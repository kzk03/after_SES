"""D4: リポジトリ単位の AI レビュー導入をイベントとしたイベントスタディ。

D3(モジュール単位の「AI レビュー初参入月」を処置とする設計)は失敗した。
処置時点が定義上「活動が活発な月」に選ばれるため、k=0 で全アウトカムが跳ね上がり
(log 人間評価件数 +1.31, is_active +0.55)、イベントスタディとして成立しなかった。

D4 では処置をリポジトリ単位に上げる。AI レビューボットの有効化はリポジトリ管理者の
判断であり、個々のモジュールがその月に活発かどうかとは独立に決まる。
D2 の比率推移を見ると、backstage(2026-01: 1%→19%→43%)、grafana(2026-07: 6%→25%)、
NVIDIA/Megatron-LM(2026-03: 0%→16%)のように明確な段差を持つリポジトリがある。

処置の定義:
  g_r = 「直後3ヶ月平均の AI レビュー比率が、直前3ヶ月平均より JUMP 以上高く、
          かつ直後3ヶ月平均が LEVEL 以上」を最初に満たす月
  - never-treated: 全期間で AI レビュー比率が LEVEL 未満 → 対照群
  - always-treated: 観測開始時点で既に LEVEL 以上 → 段差前が観測できないので除外

単位は module x month(バランスドパネル)、固定効果は module + month、
標準誤差はリポジトリクラスタ(処置の割り当て単位)。

実行: uv run python pilot/phase2/scripts/d4_repo_event_study.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from common import OUT_DIR, build_count_panel

JUMP = 0.05   # 導入と認める比率の段差
LEVEL = 0.10  # 導入後に到達すべき水準
WIN = 3       # 段差判定に使う前後の月数
LEAD_LAG = 3
REF = -1


def detect_adoption(df: pd.DataFrame) -> tuple[dict[str, str], list[str], list[str]]:
    """リポジトリごとの AI レビュー導入月を検出する。

    returns: (adoption {repo: month}, never_treated, always_treated)
    """
    by = df.groupby(["repo", "month"])[["n_E_agent", "n_E_human"]].sum().reset_index()
    by["share"] = by["n_E_agent"] / (by["n_E_agent"] + by["n_E_human"])
    months = sorted(df["month"].unique())

    adoption: dict[str, str] = {}
    never: list[str] = []
    always: list[str] = []
    for repo, g in by.groupby("repo"):
        s = g.set_index("month")["share"].reindex(months)
        # 観測開始直後(最初のWIN月)が既に高水準なら always-treated
        head = s.iloc[:WIN].mean()
        if pd.notna(head) and head >= LEVEL:
            always.append(repo)
            continue
        found = None
        for i in range(1, len(months)):
            pre = s.iloc[max(0, i - WIN):i].mean()
            post = s.iloc[i:i + WIN].mean()
            if pd.isna(pre) or pd.isna(post):
                continue
            if post - pre >= JUMP and post >= LEVEL:
                found = months[i]
                break
        if found:
            adoption[repo] = found
        elif s.max(skipna=True) < LEVEL:
            never.append(repo)
        # 上記どちらでもない(中途半端に上下する)リポジトリは処置群にも対照群にも入れない
    return adoption, never, always


def build_balanced(df: pd.DataFrame) -> pd.DataFrame:
    """module x month のバランスドパネルを作る(イベントが無い月は0件として埋める)。

    ただし0で埋めてよいのは「そのリポジトリを観測できている期間」だけ。
    収集期間がリポジトリごとに違う場合(巨大リポジトリで 2023-01 まで遡りきれなかった等)、
    データが無いだけの月を「活動ゼロ」と誤認すると、イベントスタディの前期に
    見せかけの落ち込みが入る。観測範囲外は行ごと落とす。
    """
    months = sorted(df["month"].unique())
    modules = df[["module_id_global", "repo"]].drop_duplicates()
    grid = modules.merge(pd.DataFrame({"month": months}), how="cross")
    span = df.groupby("repo")["month"].agg(obs_min="min", obs_max="max")
    grid = grid.join(span, on="repo")
    grid = grid[(grid["month"] >= grid["obs_min"]) & (grid["month"] <= grid["obs_max"])]
    grid = grid.drop(columns=["obs_min", "obs_max"])
    count_cols = [c for c in df.columns if c.startswith("n_")]
    bal = grid.merge(df[["module_id_global", "month", *count_cols]], on=["module_id_global", "month"], how="left")
    bal[count_cols] = bal[count_cols].fillna(0.0)
    bal["month_period"] = pd.PeriodIndex(bal["month"], freq="M").to_timestamp()
    bal["month_idx"] = bal["month"].map({m: i for i, m in enumerate(months)})
    return bal


def fit(bal: pd.DataFrame, y: str, cols: list[str]) -> dict:
    d = bal.dropna(subset=[y]).set_index(["module_id_global", "month_period"])
    base = PanelOLS(d[y], d[cols], entity_effects=True, time_effects=True, drop_absorbed=True)
    r_r = base.fit(cov_type="clustered", clusters=d["repo"])
    r_m = PanelOLS(d[y], d[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    return {
        "y": y,
        "n_obs": int(r_r.nobs),
        "n_repos": int(d["repo"].nunique()),
        "params": r_r.params.to_dict(),
        "se_repo": r_r.std_errors.to_dict(),
        "p_repo": r_r.pvalues.to_dict(),
        "p_module": r_m.pvalues.to_dict(),
    }


def print_table(r: dict, cols: list[str]) -> None:
    print(f"\n--- {r['y']} (n={r['n_obs']}, repos={r['n_repos']}) ---")
    print(f"{'k':>5} {'coef':>10} {'se(repo)':>10} {'p(repo)':>10} {'p(mod)':>10}")
    for c in cols:
        if c not in r["params"]:
            continue
        k = ("-" + c[3:]) if c.startswith("k_m") else ("+" + c[3:])
        star = " *" if r["p_repo"][c] < 0.05 else ""
        print(f"{k:>5} {r['params'][c]:>+10.4f} {r['se_repo'][c]:>10.4f} "
              f"{r['p_repo'][c]:>10.4g} {r['p_module'][c]:>10.4g}{star}")
    print(f"{'-1':>5} {'0 (ref)':>10}")


def main() -> None:
    df = build_count_panel()
    adoption, never, always = detect_adoption(df)

    print("=== リポジトリ単位の AI レビュー導入検出 ===")
    print(f"treated ({len(adoption)}):")
    for r, m in sorted(adoption.items(), key=lambda kv: kv[1]):
        print(f"  {m}  {r}")
    print(f"never-treated ({len(never)}): {', '.join(sorted(never))}")
    print(f"always-treated (除外, {len(always)}): {', '.join(sorted(always))}")
    ambiguous = set(df["repo"]) - set(adoption) - set(never) - set(always)
    print(f"判定不能 (除外, {len(ambiguous)}): {', '.join(sorted(ambiguous))}")

    if len(adoption) < 3:
        print("\n処置リポジトリが少なすぎる。閾値を見直すこと。")
        return

    bal = build_balanced(df)
    # 対照群は never-treated のみ。always-treated と判定不能は落とす
    keep_repos = set(adoption) | set(never)
    bal = bal[bal["repo"].isin(keep_repos)].copy()

    months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(months)}
    bal["g"] = bal["repo"].map({r: midx[m] for r, m in adoption.items()})
    bal["event_time"] = bal["month_idx"] - bal["g"]

    cols: list[str] = []
    for j in range(-LEAD_LAG, LEAD_LAG + 1):
        if j == REF:
            continue
        name = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
        k = bal["event_time"]
        if j == -LEAD_LAG:
            val = k <= j
        elif j == LEAD_LAG:
            val = k >= j
        else:
            val = k == j
        bal[name] = val.fillna(False).astype(float)
        cols.append(name)

    bal["log_n_E_human"] = np.log1p(bal["n_E_human"])
    bal["log_n_E_agent"] = np.log1p(bal["n_E_agent"])
    bal["log_n_E"] = np.log1p(bal["n_E"])
    bal["log_n_E_human_actors"] = np.log1p(bal["n_E_human_actors"])
    bal["log_n_G_human"] = np.log1p(bal["n_G_human_strict"])
    bal["H_E"] = np.where(bal["n_E"] > 0, bal["n_E_human"] / bal["n_E"], np.nan)
    bal["e_human_per_pr"] = np.where(bal["n_E_prs"] > 0, bal["n_E_human"] / bal["n_E_prs"], np.nan)
    bal["e_agent_per_pr"] = np.where(bal["n_E_prs"] > 0, bal["n_E_agent"] / bal["n_E_prs"], np.nan)
    bal["is_active"] = (bal["n_E"] + bal["n_G"] > 0).astype(float)

    out: dict = {
        "spec": {"jump": JUMP, "level": LEVEL, "window": WIN, "lead_lag": LEAD_LAG},
        "adoption": adoption,
        "never_treated": never,
        "always_treated_excluded": always,
        "ambiguous_excluded": sorted(ambiguous),
        "n_modules": int(bal["module_id_global"].nunique()),
        "models": {},
    }

    print("\n\n=== イベントスタディ(処置=リポジトリの AI レビュー導入) ===")
    print("first stage(AI レビューが実際に増えたか)と、人間側アウトカムを並べて見る")
    for y in [
        "log_n_E_agent",      # first stage
        "e_agent_per_pr",     # first stage(PRあたり)
        "log_n_E_human",      # ★人間のレビュー量は減るか
        "log_n_E_human_actors",
        "e_human_per_pr",     # ★PRあたりの人間レビューは減るか
        "H_E",
        "log_n_G_human",
        "is_active",
    ]:
        r = fit(bal, y, cols)
        out["models"][y] = r
        print_table(r, cols)

    (OUT_DIR / "d4_repo_event_study.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd4_repo_event_study.json'}")


if __name__ == "__main__":
    main()

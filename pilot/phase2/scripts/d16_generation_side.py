"""D16: 生成工程でも同じ構造か — AI は書く側でも人間を置き換えていないのか。

ここまでは評価工程(レビュー)を見てきた。もし生成工程(コードを書く)でも
「人間の量は変わらず AI が上乗せされているだけ」なら、
**share 指標が AI の影響を過大に読ませる問題は、工程を問わず一般に成り立つ**ことになり、
論文の主張の射程が大きく広がる。

評価側との違いは処置の定義。評価側は「AI レビューボットの導入」だったが、
生成側は「コーディングエージェントによる PR 作成の開始」をイベントとする。
A'(生成の AI 関与度)は PR 作成者が agent か、または commit trailer に AI が残っているかで判定する
(gates.md Loop1 で確定した主指標)。

実行: uv run python pilot/phase2/scripts/d16_generation_side.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats

from common import OUT_DIR, build_count_panel
from d4_repo_event_study import LEAD_LAG, REF, build_balanced

JUMP = 0.05
LEVEL = 0.10
WIN = 3


def detect_generation_adoption(df: pd.DataFrame) -> tuple[dict[str, str], list[str], list[str]]:
    """リポジトリごとの「AI 生成の本格導入月」を、評価側と同じ基準で検出する。"""
    by = df.groupby(["repo", "month"])[["n_G_agent", "n_G_human_strict"]].sum().reset_index()
    by["share"] = by["n_G_agent"] / (by["n_G_agent"] + by["n_G_human_strict"])
    months = sorted(df["month"].unique())
    adoption, never, always = {}, [], []
    for repo, g in by.groupby("repo"):
        s = g.set_index("month")["share"].reindex(months)
        if pd.notna(s.iloc[:WIN].mean()) and s.iloc[:WIN].mean() >= LEVEL:
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
    return adoption, never, always


def fit(d: pd.DataFrame, y: str, cols: list[str]) -> dict:
    dd = d.dropna(subset=[y, *cols]).set_index(["module_id_global", "month_period"])
    r = PanelOLS(dd[y], dd[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", clusters=dd["repo"]
    )
    return {"params": r.params.to_dict(), "p_repo": r.pvalues.to_dict(),
            "se_repo": r.std_errors.to_dict(), "n_obs": int(r.nobs),
            "n_repos": int(dd["repo"].nunique())}


def main() -> None:
    df = build_count_panel()
    months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(months)}
    out: dict = {}

    # --- 1. 記述: 生成工程の月次推移 ------------------------------------------------
    m = df.groupby("month")[["n_G_human_strict", "n_G_agent", "n_G_prs"]].sum()
    m["H_G_prime"] = m["n_G_human_strict"] / (m["n_G_human_strict"] + m["n_G_agent"])
    print("=== 生成工程の月次推移(39リポジトリ合算) ===")
    print(f"{'month':<10}{'人間の生成':>12}{'AIの生成':>12}{'人間比率 H_G':>14}")
    for mon, r in m.iterrows():
        print(f"{mon:<10}{int(r['n_G_human_strict']):>12,}{int(r['n_G_agent']):>12,}{r['H_G_prime']:>14.3f}")
    first, last = m.iloc[0], m.iloc[-2]  # 最終月は収集途中なので1つ前を使う
    print(f"\n  人間の生成: {int(first['n_G_human_strict']):,} -> {int(last['n_G_human_strict']):,} "
          f"({last['n_G_human_strict']/first['n_G_human_strict']:.2f}倍)")
    print(f"  AI の生成 : {int(first['n_G_agent']):,} -> {int(last['n_G_agent']):,} "
          f"({last['n_G_agent']/first['n_G_agent']:.1f}倍)")
    print(f"  人間比率  : {first['H_G_prime']:.3f} -> {last['H_G_prime']:.3f}")
    out["monthly"] = m.reset_index().to_dict(orient="records")

    # --- 2. AI 生成の導入イベント ---------------------------------------------------
    adoption, never, always = detect_generation_adoption(df)
    print(f"\n=== 生成側の導入イベント検出 ===")
    print(f"treated ({len(adoption)}):")
    for r, mo in sorted(adoption.items(), key=lambda kv: kv[1]):
        print(f"  {mo}  {r}")
    print(f"never-treated ({len(never)})、always-treated 除外 ({len(always)}): {', '.join(sorted(always))}")
    out["adoption"] = adoption
    out["never"] = never
    out["always_excluded"] = always

    if len(adoption) < 3:
        print("\n処置リポジトリが少なすぎるため、イベントスタディは行わない。")
        (OUT_DIR / "d16_generation_side.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        return

    bal = build_balanced(df)
    bal = bal[bal["repo"].isin(set(adoption) | set(never))].copy()
    bal["g"] = bal["repo"].map({r: midx[mo] for r, mo in adoption.items()})
    k = bal["month_idx"] - bal["g"]
    cols = []
    for j in range(-LEAD_LAG, LEAD_LAG + 1):
        if j == REF:
            continue
        name = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
        val = (k <= j) if j == -LEAD_LAG else ((k >= j) if j == LEAD_LAG else (k == j))
        bal[name] = val.fillna(False).astype(float)
        cols.append(name)

    bal["log_n_G_human"] = np.log1p(bal["n_G_human_strict"])
    bal["log_n_G_agent"] = np.log1p(bal["n_G_agent"])
    bal["log_n_G_human_actors"] = np.log1p(bal["n_G_human_actors"])
    bal["H_G_prime"] = np.where(bal["n_G"] > 0, bal["n_G_human_strict"] / bal["n_G"], np.nan)
    # 生成側に AI が入ったとき、評価側(人間)がどう動くかも併せて見る
    bal["log_n_E_human"] = np.log1p(bal["n_E_human"])
    bal["e_human_per_pr"] = np.where(bal["n_E_prs"] > 0, bal["n_E_human"] / bal["n_E_prs"], np.nan)

    print("\n=== 生成側イベントスタディ ===")
    print(f"  {'被説明変数':<24}{'k=-2':>10}{'p':>8}{'k=+2':>10}{'p':>8}{'k=+3':>10}{'p':>8}{'n':>9}")
    out["models"] = {}
    for y in ["log_n_G_agent", "log_n_G_human", "log_n_G_human_actors", "H_G_prime",
              "log_n_E_human", "e_human_per_pr"]:
        r = fit(bal, y, cols)
        out["models"][y] = r
        m2, pm2 = r["params"].get("k_m2", np.nan), r["p_repo"].get("k_m2", np.nan)
        b2, p2 = r["params"].get("k_p2", np.nan), r["p_repo"].get("k_p2", np.nan)
        b3, p3 = r["params"].get("k_p3", np.nan), r["p_repo"].get("k_p3", np.nan)
        star = " *" if (p2 < 0.05 or p3 < 0.05) else ""
        print(f"  {y:<24}{m2:>+10.4f}{pm2:>8.3f}{b2:>+10.4f}{p2:>8.3f}{b3:>+10.4f}{p3:>8.3f}"
              f"{r['n_obs']:>9,}{star}")

    # --- 3. 同値性検定(片側: δ以上の減少はないか) -----------------------------------
    print("\n=== 「10%以上の減少はない」と言えるか(片側 non-inferiority) ===")
    out["no_decline"] = {}
    for y in ["log_n_G_human", "log_n_G_human_actors", "log_n_E_human"]:
        r = out["models"][y]
        for kk in ["k_p2", "k_p3"]:
            b, se = r["params"][kk], r["se_repo"][kk]
            p = float(1 - stats.norm.cdf((b + 0.10) / se))
            out["no_decline"][f"{y}_{kk}"] = p
            print(f"  {y:<24} {kk}  p={p:.4f}  -> {'言える' if p < 0.05 else '言えない'}")

    (OUT_DIR / "d16_generation_side.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd16_generation_side.json'}")


if __name__ == "__main__":
    main()

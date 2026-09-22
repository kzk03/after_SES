"""D3: イベントスタディ — AI レビュー参入の前後で人間の評価活動はどう動くか。

D1 の固定効果回帰は「AI 関与度が高いモジュールほど AI レビューが多く、人間のレビュー量は
変わらない」という相関を示した。しかし AI が入るモジュールは内生的に選ばれている
(D2: treated の median 総評価件数 30 に対し never-treated は 6.5)。
そこでモジュールごとの「AI レビュー初参入月」をイベントとして、その前後の動きを推定する。

設計:
  - 処置 g_m = そのモジュールで AI レビューが2件以上入った最初の月(単発ノイズを除外)
  - event time k = t - g_m、参照は k=-1。端は k<=-4, k>=+4 に bin
  - 全モジュール x 全月のバランスドパネル(活動ゼロの月も0件として使う。
    「AI 参入後に活動が止まった」ケースを取りこぼさないため)
  - module 固定効果 + 月固定効果、標準誤差は module / repo クラスタ
  - 既処置を対照に使う 2WFE の負の重み問題に対処するため、
    「never-treated + not-yet-treated のみ」に制限した推定も併せて行う

pre-trend(k<0 の係数)がゼロ近傍なら、処置前に差がなかった=並行トレンドの傍証になる。

実行: uv run python pilot/phase2/scripts/d3_event_study.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from common import OUT_DIR, build_count_panel

LEAD_LAG = 4  # -4 .. +4 (端点は bin)
REF = -1


def build_balanced(df: pd.DataFrame) -> pd.DataFrame:
    """module x month の完全バランスドパネルを作る(欠測月はイベント0件で埋める)。"""
    months = sorted(df["month"].unique())
    modules = df[["module_id_global", "repo"]].drop_duplicates()
    grid = modules.merge(pd.DataFrame({"month": months}), how="cross")
    count_cols = [c for c in df.columns if c.startswith("n_")]
    bal = grid.merge(df[["module_id_global", "month", *count_cols]], on=["module_id_global", "month"], how="left")
    bal[count_cols] = bal[count_cols].fillna(0.0)
    bal["month_period"] = pd.PeriodIndex(bal["month"], freq="M").to_timestamp()
    bal["month_idx"] = bal["month"].map({m: i for i, m in enumerate(months)})
    return bal


def add_event_time(bal: pd.DataFrame, treat_col: str) -> pd.DataFrame:
    """treat_col が True になる初月を処置時点として event time を付与する。"""
    hit = bal[bal[treat_col]]
    g = hit.groupby("module_id_global")["month_idx"].min().rename("g")
    bal = bal.join(g, on="module_id_global")
    bal["event_time"] = bal["month_idx"] - bal["g"]
    bal["treated_ever"] = bal["g"].notna()
    return bal


def event_dummies(bal: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """event time ダミー列を作る。never-treated は全て0。参照 k=-1 は作らない。"""
    k = bal["event_time"]
    cols: list[str] = []
    for j in range(-LEAD_LAG, LEAD_LAG + 1):
        if j == REF:
            continue
        name = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
        if j == -LEAD_LAG:
            val = (k <= j)
        elif j == LEAD_LAG:
            val = (k >= j)
        else:
            val = (k == j)
        bal[name] = val.fillna(False).astype(float)
        cols.append(name)
    return bal, cols


def fit(bal: pd.DataFrame, y: str, cols: list[str]) -> dict:
    d = bal.dropna(subset=[y]).set_index(["module_id_global", "month_period"])
    mod = PanelOLS(d[y], d[cols], entity_effects=True, time_effects=True, drop_absorbed=True)
    r_m = mod.fit(cov_type="clustered", cluster_entity=True)
    r_r = PanelOLS(d[y], d[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", clusters=d["repo"]
    )
    return {
        "y": y,
        "n_obs": int(r_m.nobs),
        "params": r_m.params.to_dict(),
        "p_module": r_m.pvalues.to_dict(),
        "se_module": r_m.std_errors.to_dict(),
        "p_repo": r_r.pvalues.to_dict(),
        "se_repo": r_r.std_errors.to_dict(),
    }


def print_event_table(r: dict, cols: list[str]) -> None:
    print(f"\n--- {r['y']} (n={r['n_obs']}) ---")
    print(f"{'k':>5} {'coef':>10} {'se(mod)':>10} {'p(mod)':>10} {'p(repo)':>10}")
    for c in cols:
        k = ("-" + c[3:]) if c.startswith("k_m") else ("+" + c[3:])
        if c not in r["params"]:
            continue
        print(f"{k:>5} {r['params'][c]:>+10.4f} {r['se_module'][c]:>10.4f} "
              f"{r['p_module'][c]:>10.4g} {r['p_repo'][c]:>10.4g}")
    print(f"{'-1':>5} {'0 (ref)':>10}")


def main() -> None:
    df = build_count_panel()
    bal = build_balanced(df)
    bal["treat_review"] = bal["n_E_agent"] >= 2
    bal = add_event_time(bal, "treat_review")
    bal, cols = event_dummies(bal)

    # アウトカム
    bal["log_n_E_human"] = np.log1p(bal["n_E_human"])
    bal["log_n_E_agent"] = np.log1p(bal["n_E_agent"])
    bal["log_n_E"] = np.log1p(bal["n_E"])
    bal["log_n_G_human"] = np.log1p(bal["n_G_human_strict"])
    bal["log_n_E_human_actors"] = np.log1p(bal["n_E_human_actors"])
    bal["H_E"] = np.where(bal["n_E"] > 0, bal["n_E_human"] / bal["n_E"], np.nan)
    bal["e_human_per_pr"] = np.where(bal["n_E_prs"] > 0, bal["n_E_human"] / bal["n_E_prs"], np.nan)
    bal["is_active"] = (bal["n_E"] + bal["n_G"] > 0).astype(float)

    outcomes = [
        "log_n_E_human",
        "log_n_E_agent",
        "log_n_E",
        "log_n_E_human_actors",
        "log_n_G_human",
        "H_E",
        "e_human_per_pr",
        "is_active",
    ]

    out: dict = {"spec": {"lead_lag": LEAD_LAG, "ref": REF, "treat": "n_E_agent>=2"},
                 "n_modules": int(bal["module_id_global"].nunique()),
                 "n_treated": int(bal.groupby("module_id_global")["treated_ever"].first().sum())}

    print("=== A. 全サンプル 2WFE イベントスタディ ===")
    out["full"] = {}
    for y in outcomes:
        r = fit(bal, y, cols)
        out["full"][y] = r
        print_event_table(r, cols)

    # --- B. not-yet-treated + never-treated のみを対照に使う -----------------------
    # 各行を残す条件: 未処置(g が無い or t < g) または 処置後ウィンドウ内(0 <= k <= LEAD_LAG)
    # = 「既に処置されて長く経ったモジュール」を対照として使わないようにする
    keep = bal["g"].isna() | (bal["event_time"] < 0) | (bal["event_time"].between(0, LEAD_LAG))
    sub = bal[keep].copy()
    print("\n\n=== B. not-yet-treated / never-treated のみを対照にした推定 ===")
    print(f"(サンプル {len(bal)} -> {len(sub)} 行)")
    out["clean_control"] = {}
    for y in ["log_n_E_human", "log_n_E_agent", "H_E", "e_human_per_pr", "is_active"]:
        r = fit(sub, y, cols)
        out["clean_control"][y] = r
        print_event_table(r, cols)

    (OUT_DIR / "d3_event_study.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd3_event_study.json'}")


if __name__ == "__main__":
    main()

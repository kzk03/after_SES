"""D10: メカニズム — AI が指摘を肩代わりすると人間の指摘は減るのか。

D4/D7/D8 で「AI レビュー導入後も人間側の指標は動かない」ことが分かった。
では AI と人間は同じ仕事をしていないのか。GitHub のレビューは state を持つので、
それを手がかりにする。

  CHANGES_REQUESTED : 修正を要求する(指摘)
  APPROVED          : 承認する(判断)
  COMMENTED         : 意見を述べるだけ

「AI が先に指摘するので人間は指摘しなくてよくなる」なら、導入後に
人間の CHANGES_REQUESTED が減るはず。減らないなら、AI と人間は別の層を見ている。
また AI が APPROVED を出しているかどうかは「AI に判断権限があるか」を示す。

実行: uv run python pilot/phase2/scripts/d10_review_kind.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from common import OUT_DIR, build_count_panel
from d4_repo_event_study import LEAD_LAG, REF, detect_adoption


def fit(d: pd.DataFrame, y: str, cols: list[str]) -> dict:
    dd = d.dropna(subset=[y, *cols]).set_index(["module_id_global", "month_period"])
    if dd.empty or dd["repo"].nunique() < 5:
        return {}
    r_r = PanelOLS(dd[y], dd[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", clusters=dd["repo"]
    )
    r_m = PanelOLS(dd[y], dd[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    return {
        "y": y, "n_obs": int(r_r.nobs), "n_repos": int(dd["repo"].nunique()),
        "params": r_r.params.to_dict(), "se_repo": r_r.std_errors.to_dict(),
        "p_repo": r_r.pvalues.to_dict(), "p_module": r_m.pvalues.to_dict(),
    }


def print_table(r: dict, ev_cols: list[str]) -> None:
    if not r:
        print("  (推定できず)")
        return
    print(f"\n--- {r['y']} (n={r['n_obs']}, repos={r['n_repos']}) ---")
    print(f"{'k':>5} {'coef':>10} {'se(repo)':>10} {'p(repo)':>10} {'p(mod)':>10}")
    for c in ev_cols:
        if c not in r["params"]:
            continue
        k = ("-" + c[3:]) if c.startswith("k_m") else ("+" + c[3:])
        star = " *" if r["p_repo"][c] < 0.05 else ""
        print(f"{k:>5} {r['params'][c]:>+10.4f} {r['se_repo'][c]:>10.4f} "
              f"{r['p_repo'][c]:>10.4g} {r['p_module'][c]:>10.4g}{star}")
    print(f"{'-1':>5} {'0 (ref)':>10}")


def main() -> None:
    pr = pd.read_parquet(OUT_DIR / "d5_pr_level.parquet")

    # --- 記述: 誰がどの種類のレビューを出しているか ---------------------------------
    tot = {
        "human": {
            "APPROVED": int(pr["n_human_approved"].sum()),
            "CHANGES_REQUESTED": int(pr["n_human_changes_requested"].sum()),
            "COMMENTED": int(pr["n_human_commented"].sum()),
        },
        "agent": {
            "APPROVED": int(pr["n_ai_approved"].sum()),
            "CHANGES_REQUESTED": int(pr["n_ai_changes_requested"].sum()),
            "COMMENTED": int(pr["n_ai_commented"].sum()),
        },
    }
    print("=== レビュー種別の分布(全 PR) ===")
    for cat, d in tot.items():
        s = sum(d.values())
        print(f"  {cat:<6} 合計 {s:>8,}  " + "  ".join(f"{k}={v:,}({v/s:.1%})" for k, v in d.items()))

    df = build_count_panel()
    months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(months)}
    adoption, never, always = detect_adoption(df)

    agg = (
        pr.groupby(["module_id_global", "repo", "month"])
        .agg(
            n_prs=("pr_number", "count"),
            n_human_cr=("n_human_changes_requested", "sum"),
            n_human_ap=("n_human_approved", "sum"),
            n_human_co=("n_human_commented", "sum"),
            n_ai_cr=("n_ai_changes_requested", "sum"),
            n_ai_ap=("n_ai_approved", "sum"),
            n_ai_co=("n_ai_commented", "sum"),
            avg_pr_lines=("pr_lines", "mean"),
        )
        .reset_index()
    )
    agg = agg[agg["repo"].isin(set(adoption) | set(never))].copy()
    agg["month_period"] = pd.PeriodIndex(agg["month"], freq="M").to_timestamp()
    agg["month_idx"] = agg["month"].map(midx)
    agg = agg.dropna(subset=["month_idx"])
    agg["g"] = agg["repo"].map({r: midx[m] for r, m in adoption.items()})
    agg["event_time"] = agg["month_idx"] - agg["g"]

    n_human = agg["n_human_cr"] + agg["n_human_ap"] + agg["n_human_co"]
    agg["human_cr_share"] = np.where(n_human > 0, agg["n_human_cr"] / n_human, np.nan)
    agg["human_ap_share"] = np.where(n_human > 0, agg["n_human_ap"] / n_human, np.nan)
    agg["log_n_human_cr"] = np.log1p(agg["n_human_cr"])
    agg["log_n_human_ap"] = np.log1p(agg["n_human_ap"])
    agg["log_n_ai_cr"] = np.log1p(agg["n_ai_cr"])
    # 1 PR あたりに直した指標(スループットの影響を除く)
    agg["human_cr_per_pr"] = agg["n_human_cr"] / agg["n_prs"]
    agg["ai_cr_per_pr"] = agg["n_ai_cr"] / agg["n_prs"]

    ev_cols: list[str] = []
    for j in range(-LEAD_LAG, LEAD_LAG + 1):
        if j == REF:
            continue
        name = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
        k = agg["event_time"]
        val = (k <= j) if j == -LEAD_LAG else ((k >= j) if j == LEAD_LAG else (k == j))
        agg[name] = val.fillna(False).astype(float)
        ev_cols.append(name)

    out: dict = {"state_distribution": tot, "adoption": adoption, "models": {}}
    print("\n\n=== レビュー種別のイベントスタディ(処置=AI レビュー導入) ===")
    for y in [
        "log_n_ai_cr",        # first stage: AI の指摘は増えたか
        "ai_cr_per_pr",
        "log_n_human_cr",     # ★人間の指摘は減ったか
        "human_cr_per_pr",    # ★1 PR あたりの人間の指摘
        "human_cr_share",     # 人間のレビューに占める指摘の割合
        "log_n_human_ap",     # 人間の承認
        "human_ap_share",
    ]:
        r = fit(agg, y, ev_cols)
        out["models"][y] = r
        print_table(r, ev_cols)

    (OUT_DIR / "d10_review_kind.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd10_review_kind.json'}")


if __name__ == "__main__":
    main()

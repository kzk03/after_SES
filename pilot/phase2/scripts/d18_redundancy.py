"""D18: AI レビューは「冗長な上乗せ」か「人手の届かない所の補完」か。

D4/D14/D15 で、AI レビュー導入後も人間のレビュー量は総量でも PR あたりでも分配でも変わらない
ことが分かった。ではその AI レビューはどこに付いているのか。2つの可能性がある。

  (a) 冗長(redundant): 人間が既にレビューする PR に重ねて付いている
      → 人間の作業は減らず、読む量だけ増える
  (b) 補完(complementary): 人間の手が回らない PR をカバーしている
      → 人間の作業は減らないが、レビューされる PR の範囲は広がる(価値がある)

(b) なら「人間は減らないが AI には意味がある」と言えるので、主張が変わる。
  - レビューが1件も付かない PR の割合(レビュー空白)が導入後に減るか
  - AI レビューが付いた PR のうち、人間レビューも付いている割合
  - 人間レビュー0件だが AI レビューはある PR の割合
を見る。

実行: uv run python pilot/phase2/scripts/d18_redundancy.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from common import INTERIM_DIR, OUT_DIR, build_count_panel, load_repos, repo_dir_name
from d4_repo_event_study import LEAD_LAG, REF, detect_adoption


def build_pr_coverage() -> pd.DataFrame:
    """全 PR について、人間 / AI のレビュー有無を持つ表を作る。

    d5_pr_level.parquet はレビューがある PR だけでなく全 PR を含むので、
    「レビューが1件も付かなかった PR」も数えられる。
    """
    pr = pd.read_parquet(OUT_DIR / "d5_pr_level.parquet")
    pr["has_human_review"] = pr["n_human_reviews_total"] > 0
    pr["has_ai_review"] = pr["n_ai_reviews_total"] > 0
    pr["no_review_at_all"] = ~(pr["has_human_review"] | pr["has_ai_review"])
    pr["ai_only"] = pr["has_ai_review"] & ~pr["has_human_review"]
    pr["human_only"] = pr["has_human_review"] & ~pr["has_ai_review"]
    pr["both"] = pr["has_human_review"] & pr["has_ai_review"]
    return pr


def fit(d: pd.DataFrame, y: str, cols: list[str]) -> dict:
    dd = d.dropna(subset=[y, *cols]).set_index(["module_id_global", "month_period"])
    r = PanelOLS(dd[y], dd[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", clusters=dd["repo"]
    )
    return {"params": r.params.to_dict(), "p_repo": r.pvalues.to_dict(),
            "se_repo": r.std_errors.to_dict(), "n_obs": int(r.nobs)}


def main() -> None:
    pr = build_pr_coverage()
    df = build_count_panel()
    months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(months)}
    adoption, never, always = detect_adoption(df)
    out: dict = {}

    # --- 1. 記述: PR のレビュー被覆 ------------------------------------------------
    n = len(pr)
    print(f"=== PR のレビュー被覆(全 {n:,} PR) ===")
    for label, col in [("人間のみ", "human_only"), ("人間+AI", "both"),
                       ("AIのみ", "ai_only"), ("レビューなし", "no_review_at_all")]:
        print(f"  {label:<12}{pr[col].sum():>9,}  ({pr[col].mean():>6.1%})")
    reviewed_by_ai = pr[pr["has_ai_review"]]
    print(f"\n  AI レビューが付いた PR {len(reviewed_by_ai):,} 件のうち、"
          f"**{reviewed_by_ai['has_human_review'].mean():.1%}** は人間もレビューしている")
    out["coverage"] = {c: {"n": int(pr[c].sum()), "share": float(pr[c].mean())}
                       for c in ["human_only", "both", "ai_only", "no_review_at_all"]}
    out["ai_reviewed_prs_also_human_reviewed"] = float(reviewed_by_ai["has_human_review"].mean())

    # 導入前後でこの割合が変わるか(処置リポジトリのみ、素の比較)
    pr["month_idx"] = pr["month"].map(midx)
    pr["g"] = pr["repo"].map({r: midx[m] for r, m in adoption.items()})
    treated = pr[pr["g"].notna()].copy()
    treated["post"] = treated["month_idx"] >= treated["g"]
    print("\n  処置リポジトリでの導入前後(素の比較):")
    print(f"  {'':12}{'AIのみ':>10}{'人間+AI':>10}{'人間のみ':>10}{'レビューなし':>14}")
    for post, g in treated.groupby("post"):
        lab = "導入後" if post else "導入前"
        print(f"  {lab:<12}{g['ai_only'].mean():>10.1%}{g['both'].mean():>10.1%}"
              f"{g['human_only'].mean():>10.1%}{g['no_review_at_all'].mean():>14.1%}")
        out[f"treated_{'post' if post else 'pre'}"] = {
            c: float(g[c].mean()) for c in ["ai_only", "both", "human_only", "no_review_at_all"]}

    # --- 2. モジュール x 月に集計してイベントスタディ --------------------------------
    agg = (
        pr.groupby(["module_id_global", "repo", "month"])
        .agg(n_prs=("pr_number", "count"),
             share_no_review=("no_review_at_all", "mean"),
             share_ai_only=("ai_only", "mean"),
             share_both=("both", "mean"),
             share_human_any=("has_human_review", "mean"))
        .reset_index()
    )
    agg = agg[agg["repo"].isin(set(adoption) | set(never))].copy()
    agg["month_period"] = pd.PeriodIndex(agg["month"], freq="M").to_timestamp()
    agg["month_idx"] = agg["month"].map(midx)
    agg["g"] = agg["repo"].map({r: midx[m] for r, m in adoption.items()})
    k = agg["month_idx"] - agg["g"]
    cols = []
    for j in range(-LEAD_LAG, LEAD_LAG + 1):
        if j == REF:
            continue
        nm = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
        val = (k <= j) if j == -LEAD_LAG else ((k >= j) if j == LEAD_LAG else (k == j))
        agg[nm] = val.fillna(False).astype(float)
        cols.append(nm)

    print("\n=== イベントスタディ(処置=AI レビュー導入) ===")
    print(f"  {'被説明変数':<22}{'k=-2':>10}{'p':>8}{'k=+2':>10}{'p':>8}{'k=+3':>10}{'p':>8}")
    out["models"] = {}
    labels = {
        "share_no_review": "レビュー0件のPR割合",
        "share_ai_only": "AIのみレビューのPR割合",
        "share_both": "人間+AIのPR割合",
        "share_human_any": "人間がレビューしたPR割合",
    }
    for y, lab in labels.items():
        r = fit(agg, y, cols)
        out["models"][y] = r
        m2, pm2 = r["params"].get("k_m2", np.nan), r["p_repo"].get("k_m2", np.nan)
        b2, p2 = r["params"].get("k_p2", np.nan), r["p_repo"].get("k_p2", np.nan)
        b3, p3 = r["params"].get("k_p3", np.nan), r["p_repo"].get("k_p3", np.nan)
        star = " *" if (p2 < 0.05 or p3 < 0.05) else ""
        print(f"  {lab:<22}{m2:>+10.4f}{pm2:>8.3f}{b2:>+10.4f}{p2:>8.3f}{b3:>+10.4f}{p3:>8.3f}{star}")

    (OUT_DIR / "d18_redundancy.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd18_redundancy.json'}")


if __name__ == "__main__":
    main()

"""D7: AI レビュー導入が人間の負担と PR の帰結に与える影響(イベントスタディ)。

D4 で「AI レビュー導入後も人間のレビュー量は減らない」ことが分かった。
では人間側に何が起きているのか。RQ3 の相関分析は「AI 関与度が高いモジュールほど
承認前に読む AI コメントが多く、承認までの日数が長い」を示していたが、
これを D4 と同じ因果識別(リポジトリ単位の導入イベント)で確認する。

アウトカムは D5 で作った PR 単位データを module x month に集計したもの:
  - avg_n_ai_reviews_before_approval : 人間が承認するまでに積まれた AI コメント数(負担)
  - avg_approval_lag_days            : 人間の承認までの日数
  - avg_time_to_merge_days           : マージまでの日数
  - approval_rate / merge_rate       : 承認・マージに至った割合(品質と選択の代理)
  - avg_n_review_rounds              : レビュー往復の総数

PR 規模(行数・ファイル数)は統制変数として入れた版も出す。

実行: uv run python pilot/phase2/scripts/d7_burden_event_study.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from common import OUT_DIR, build_count_panel
from d4_repo_event_study import LEAD_LAG, REF, build_balanced, detect_adoption


def build_pr_aggregates() -> pd.DataFrame:
    pr = pd.read_parquet(OUT_DIR / "d5_pr_level.parquet")
    agg = (
        pr.groupby(["module_id_global", "repo", "month"])
        .agg(
            n_prs=("pr_number", "count"),
            approval_rate=("has_human_approval", "mean"),
            merge_rate=("merged", "mean"),
            avg_approval_lag_days=("approval_lag_days", "mean"),
            avg_time_to_merge_days=("time_to_merge_days", "mean"),
            avg_ai_before_approval=("n_ai_reviews_before_approval", "mean"),
            avg_human_before_approval=("n_human_reviews_before_approval", "mean"),
            avg_review_rounds=("n_review_rounds", "mean"),
            avg_pr_lines=("pr_lines", "mean"),
            avg_pr_files=("pr_files", "mean"),
        )
        .reset_index()
    )
    agg["log_ai_before_approval"] = np.log1p(agg["avg_ai_before_approval"])
    agg["log_human_before_approval"] = np.log1p(agg["avg_human_before_approval"])
    agg["log_pr_lines"] = np.log1p(agg["avg_pr_lines"])
    agg["log_pr_files"] = np.log1p(agg["avg_pr_files"])
    agg["log_n_prs"] = np.log1p(agg["n_prs"])
    return agg


def fit(d: pd.DataFrame, y: str, cols: list[str]) -> dict:
    dd = d.dropna(subset=[y, *cols]).set_index(["module_id_global", "month_period"])
    if dd.empty:
        return {}
    r_r = PanelOLS(dd[y], dd[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", clusters=dd["repo"]
    )
    r_m = PanelOLS(dd[y], dd[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    return {
        "y": y,
        "n_obs": int(r_r.nobs),
        "n_repos": int(dd["repo"].nunique()),
        "params": r_r.params.to_dict(),
        "se_repo": r_r.std_errors.to_dict(),
        "p_repo": r_r.pvalues.to_dict(),
        "p_module": r_m.pvalues.to_dict(),
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
    df = build_count_panel()
    adoption, never, always = detect_adoption(df)
    months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(months)}

    agg = build_pr_aggregates()
    agg = agg[agg["repo"].isin(set(adoption) | set(never))].copy()
    agg["month_period"] = pd.PeriodIndex(agg["month"], freq="M").to_timestamp()
    agg["month_idx"] = agg["month"].map(midx)
    agg = agg.dropna(subset=["month_idx"])
    agg["g"] = agg["repo"].map({r: midx[m] for r, m in adoption.items()})
    agg["event_time"] = agg["month_idx"] - agg["g"]

    ev_cols: list[str] = []
    for j in range(-LEAD_LAG, LEAD_LAG + 1):
        if j == REF:
            continue
        name = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
        k = agg["event_time"]
        val = (k <= j) if j == -LEAD_LAG else ((k >= j) if j == LEAD_LAG else (k == j))
        agg[name] = val.fillna(False).astype(float)
        ev_cols.append(name)

    out: dict = {"adoption": adoption, "n_never": len(never), "models": {}, "models_ctrl": {}}

    print("=== 人間の負担・PR の帰結(処置=リポジトリの AI レビュー導入) ===")
    outcomes = [
        "log_ai_before_approval",     # ★人間が承認前に読む AI コメント
        "log_human_before_approval",
        "avg_approval_lag_days",      # ★承認までの日数
        "avg_time_to_merge_days",
        "avg_review_rounds",
        "approval_rate",
        "merge_rate",
    ]
    for y in outcomes:
        r = fit(agg, y, ev_cols)
        out["models"][y] = r
        print_table(r, ev_cols)

    print("\n\n=== PR 規模(行数・ファイル数・PR数)を統制した場合 ===")
    ctrl = ev_cols + ["log_pr_lines", "log_pr_files", "log_n_prs"]
    for y in ["log_ai_before_approval", "avg_approval_lag_days", "avg_time_to_merge_days", "merge_rate"]:
        r = fit(agg, y, ctrl)
        out["models_ctrl"][y] = r
        print_table(r, ev_cols)

    (OUT_DIR / "d7_burden_event_study.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd7_burden_event_study.json'}")


if __name__ == "__main__":
    main()

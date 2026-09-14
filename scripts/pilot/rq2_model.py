"""RQ2主解析: 離脱率・新規参入 ~ 分離度(S1_EG_prime) + controls + module_FE + time_FE。

研究設計書§5.3のモデルをパネル固定効果回帰(PanelOLS)で実測する(予測ではなく計測)。

実行: uv run python scripts/pilot/rq2_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "pilot" / "processed"


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def build_controls(repo: str) -> pd.DataFrame:
    panel = pd.read_parquet(PROCESSED_DIR / repo_dir_name(repo) / "panel.parquet")
    return panel[["module_id", "month", "H_G_n", "H_E_n"]].rename(columns={"H_G_n": "n_g", "H_E_n": "n_e"})


def run_model(df: pd.DataFrame, dep_col: str, k: int) -> dict:
    sub = df[df["k"] == k].copy()
    sub = sub.dropna(subset=[dep_col, "S1_EG_prime", "n_activity"])
    sub["log_n_activity"] = np.log1p(sub["n_activity"])
    sub["module_id_global"] = sub["repo"] + "::" + sub["module_id"]
    sub["month_period"] = pd.PeriodIndex(sub["month"], freq="M").to_timestamp()
    sub = sub.set_index(["module_id_global", "month_period"])

    exog = sub[["S1_EG_prime", "log_n_activity"]]
    endog = sub[dep_col]
    mod = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    n_repo_clusters = sub.reset_index()["repo"].nunique() if "repo" in sub.columns else sub.reset_index()["module_id_global"].str.split("::").str[0].nunique()
    result = {
        "k": k,
        "dep_var": dep_col,
        "n_obs": int(res.nobs),
        "n_repo_clusters": int(n_repo_clusters),
        "params": res.params.to_dict(),
        "pvalues(cluster=module)": res.pvalues.to_dict(),
        "std_errors(cluster=module)": res.std_errors.to_dict(),
        "rsquared_within": res.rsquared_within,
    }
    if n_repo_clusters >= 5:
        repo_series = sub.reset_index()["repo"]
        repo_series.index = sub.index
        mod2 = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
        res2 = mod2.fit(cov_type="clustered", clusters=repo_series)
        result["pvalues(cluster=repo)"] = res2.pvalues.to_dict()
        result["std_errors(cluster=repo)"] = res2.std_errors.to_dict()
    return result


def main() -> None:
    outcomes = pd.read_parquet(PROCESSED_DIR / "rq2_outcomes.parquet")

    controls_frames = []
    for repo in outcomes["repo"].unique():
        c = build_controls(repo)
        c["repo"] = repo
        controls_frames.append(c)
    controls = pd.concat(controls_frames, ignore_index=True)
    controls["n_activity"] = controls["n_g"].fillna(0) + controls["n_e"].fillna(0)

    df = outcomes.merge(controls[["module_id", "month", "repo", "n_activity"]], on=["module_id", "month", "repo"], how="left")

    results = {}
    for k in sorted(df["k"].unique()):
        results[f"churn_rate_k{k}"] = run_model(df, "churn_rate", k)
        results[f"n_new_entrants_k{k}"] = run_model(df, "n_new_entrants", k)

    out_path = PROCESSED_DIR / "rq2_model_summary.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    for name, r in results.items():
        print(f"=== {name} ===")
        print(json.dumps(r, ensure_ascii=False, indent=2, default=float))
        print()
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()

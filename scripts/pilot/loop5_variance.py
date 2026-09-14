"""Loop 5 — 分散の確認(★最重要ゲート)。

processed/panel.parquet の A(m,t) について、モジュール間の分散を定量化する。
  - モジュール間の標準偏差
  - 上位25%と下位25%のAの差
  - A>0 のモジュールの割合

実行: uv run python scripts/pilot/loop5_variance.py <owner>/<repo>
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "pilot" / "processed"
REPORTS_DIR = REPO_ROOT / "pilot" / "reports"


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: loop5_variance.py <owner>/<repo>")
        sys.exit(1)
    repo = sys.argv[1]
    processed_d = PROCESSED_DIR / repo_dir_name(repo)
    panel = pd.read_parquet(processed_d / "panel.parquet")

    def variance_stats(col: str) -> dict:
        module_avg = panel.groupby("module_id")[col].mean().dropna()
        if len(module_avg) == 0:
            return {"n_modules": 0}
        q25, q75 = module_avg.quantile([0.25, 0.75])
        monthly_std = panel.groupby("month")[col].std().dropna()
        return {
            "n_modules": len(module_avg),
            "std": float(module_avg.std()),
            "q25": float(q25),
            "q75": float(q75),
            "iqr_diff": float(q75 - q25),
            "pct_positive": float((module_avg > 0).mean()),
            "monthly_std_mean": float(monthly_std.mean()) if len(monthly_std) else None,
        }, module_avg, monthly_std

    stats_a, module_avg, monthly_std = variance_stats("A")
    stats_a_prime = variance_stats("A_prime")[0] if "A_prime" in panel.columns else {}

    result = {
        "repo": repo,
        "A": stats_a,
        "A_prime(trailer込み感度分析)": stats_a_prime,
    }
    (processed_d / "loop5_variance_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 図: モジュール別Aのヒストグラム + 月次標準偏差の推移
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(module_avg, bins=30, color="#4C72B0")
    axes[0].set_title(f"{repo}\nモジュール別 A(全期間平均) の分布")
    axes[0].set_xlabel("A = 1 - H_G")
    axes[0].set_ylabel("モジュール数")

    monthly_std_sorted = monthly_std.sort_index()
    axes[1].plot(monthly_std_sorted.index, monthly_std_sorted.values, marker="o", color="#DD8452")
    axes[1].set_title("月ごとのモジュール間 A 標準偏差")
    axes[1].tick_params(axis="x", rotation=90)

    fig.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = REPORTS_DIR / f"loop5_variance_{repo_dir_name(repo)}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    print(f"=== {repo} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"図 -> {fig_path}")


if __name__ == "__main__":
    main()

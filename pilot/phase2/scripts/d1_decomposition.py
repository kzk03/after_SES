"""D1: H_E 低下の分解 — 「人間の退出」か「AI の上乗せ」か。

パイロットの主解析は H_E(m,t) = 人間の評価件数 / (人間+AI の評価件数) が
A'(m,t) の上昇とともに下がる(H1b支持)ことを示した。しかし比率が下がる理由は2つある:

  (a) displacement: 分子(人間の評価件数)が減っている  → 「人間が評価工程から退いている」
  (b) dilution:     分母(総評価件数)が膨らんでいる     → 「人間はそのまま、AI が上乗せされた」

(a) と (b) では論文の主張が正反対になるため、比率を分子・分母に分解して推定する。

実行: uv run python pilot/phase2/scripts/d1_decomposition.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import OUT_DIR, build_count_panel, fit_panel, fmt_result


def main() -> None:
    df = build_count_panel()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 0. 記述統計: 全リポジトリ合算の月次推移 ---------------------------------
    monthly = (
        df.groupby("month")[["n_G_human_strict", "n_G_agent", "n_E_human", "n_E_agent", "n_E_prs"]]
        .sum()
        .reset_index()
    )
    monthly["H_E_aggregate"] = monthly["n_E_human"] / (monthly["n_E_human"] + monthly["n_E_agent"])
    monthly["e_human_per_pr"] = monthly["n_E_human"] / monthly["n_E_prs"]
    monthly["e_agent_per_pr"] = monthly["n_E_agent"] / monthly["n_E_prs"]
    monthly.to_csv(OUT_DIR / "d1_monthly_trend.csv", index=False)

    # --- 1. 派生変数 -------------------------------------------------------------
    df["log_n_E_human"] = np.log1p(df["n_E_human"])
    df["log_n_E_agent"] = np.log1p(df["n_E_agent"])
    df["log_n_E"] = np.log1p(df["n_E"])
    df["log_n_E_human_actors"] = np.log1p(df["n_E_human_actors"])
    df["log_n_G"] = np.log1p(df["n_G"])
    df["log_n_G_prs"] = np.log1p(df["n_G_prs"])
    # 「1 PR あたり何件の人間/AI レビューが付いたか」。
    # スループット(PR 数)の違いを機械的に割り落とすので、量の増減と混ざらない
    df["e_human_per_pr"] = np.where(df["n_E_prs"] > 0, df["n_E_human"] / df["n_E_prs"], np.nan)
    df["e_agent_per_pr"] = np.where(df["n_E_prs"] > 0, df["n_E_agent"] / df["n_E_prs"], np.nan)

    # 主解析と同じサンプル(G と E が両方観測でき、H_E と A' が定義できる module x month)
    main_sample = df.dropna(subset=["H_E", "A_prime"]).copy()

    results: dict[str, dict] = {}

    def run(tag: str, y: str, x: list[str], data: pd.DataFrame) -> None:
        results[tag] = fit_panel(data, y, x)
        print(fmt_result(results[tag], key="A_prime"))

    print("=== 1. 再現: 比率モデル(パイロット主解析の再計算) ===")
    run("M0_H_E", "H_E", ["A_prime"], main_sample)

    print("\n=== 2. 分解: 分子と分母を別々に ===")
    run("M1_log_human_E", "log_n_E_human", ["A_prime"], main_sample)
    run("M2_log_agent_E", "log_n_E_agent", ["A_prime"], main_sample)
    run("M3_log_total_E", "log_n_E", ["A_prime"], main_sample)
    run("M4_log_human_reviewers", "log_n_E_human_actors", ["A_prime"], main_sample)

    print("\n=== 3. スループット統制(モジュールの PR 数を一定にした場合) ===")
    run("M5_log_human_E_ctrl", "log_n_E_human", ["A_prime", "log_n_G_prs"], main_sample)
    run("M6_log_agent_E_ctrl", "log_n_E_agent", ["A_prime", "log_n_G_prs"], main_sample)

    print("\n=== 4. PR あたりレビュー数(スループットを機械的に割り落とした指標) ===")
    run("M7_human_per_pr", "e_human_per_pr", ["A_prime"], main_sample)
    run("M8_agent_per_pr", "e_agent_per_pr", ["A_prime"], main_sample)

    print("\n=== 5. 拡張サンプル(E イベント0の月も含める。G が観測できる月すべて) ===")
    ext = df.dropna(subset=["A_prime"]).copy()
    run("M9_log_human_E_ext", "log_n_E_human", ["A_prime"], ext)
    run("M10_log_agent_E_ext", "log_n_E_agent", ["A_prime"], ext)

    summary = {
        "n_repos": int(df["repo"].nunique()),
        "n_modules": int(df["module_id_global"].nunique()),
        "main_sample_n": int(len(main_sample)),
        "monthly_trend": monthly.to_dict(orient="records"),
        "models": results,
    }
    (OUT_DIR / "d1_decomposition.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    print("\n=== 月次推移(全リポジトリ合算) ===")
    print(monthly.to_string(index=False))
    print(f"\n-> {OUT_DIR / 'd1_decomposition.json'}")


if __name__ == "__main__":
    main()

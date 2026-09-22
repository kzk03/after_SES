"""Loop 4 — 指標の算出。

events.parquet (Loop3) から module x month のパネルを作り、H_I, H_G, H_E, A を計算する。

研究設計書 §3.1 の方針に従い legacy_bot は主解析から除外する
(H_X の分母は human + agent の件数。legacy_bot はカウントしない)。
分母が0の月は欠測(NaN)。分母<3の月は low_n フラグを立てる(感度分析用)。

実行: uv run python scripts/pilot/loop4_metrics.py <owner>/<repo>
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = REPO_ROOT / "pilot" / os.environ.get("PILOT_INTERIM", "interim")
PROCESSED_DIR = REPO_ROOT / "pilot" / os.environ.get("PILOT_PROCESSED", "processed")

STAGE_TO_METRIC = {"I": "H_I", "G": "H_G", "E": "H_E"}


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: loop4_metrics.py <owner>/<repo>")
        sys.exit(1)
    repo = sys.argv[1]
    interim_d = INTERIM_DIR / repo_dir_name(repo)

    events = pd.read_parquet(interim_d / "events.parquet")
    # legacy_bot は主解析から除外(研究設計書 §3.1)
    events = events[events["actor_category"] != "legacy_bot"].copy()
    events["is_human"] = events["actor_category"] == "human"

    panels = {}
    for stage, metric_name in STAGE_TO_METRIC.items():
        sub = events[events["stage"] == stage]
        grp = sub.groupby(["module_id", "month"]).agg(n_total=("is_human", "size"), n_human=("is_human", "sum"))
        grp[metric_name] = grp["n_human"] / grp["n_total"]
        grp[f"{metric_name}_n"] = grp["n_total"]
        panels[metric_name] = grp[[metric_name, f"{metric_name}_n"]]

    panel = panels["H_I"].join(panels["H_G"], how="outer").join(panels["H_E"], how="outer")
    panel = panel.reset_index()
    panel["A"] = 1 - panel["H_G"]

    # 感度分析: commit trailer(Co-authored-by: 等)でエージェント関与を検出した場合の A'
    # (H_G は PR作成者=actor loginのみで判定しているため、人間が作成者でもAIエージェントを
    #  補助的に使ったPRを見落とす。trailer_agent_detected フラグで「人間作成 かつ trailerなし」を厳密なhumanとみなす)
    g_events = events[events["stage"] == "G"].copy()
    # trailer_agent_detected は G以外のイベント(E/I)に該当キーがないため object dtype(NaN混在)になり、
    # 素の `~` はビット反転(真偽反転にならない)になるので、astype(bool) で明示的に変換してから使う
    trailer_flag = g_events["trailer_agent_detected"].fillna(False).astype(bool)
    g_events["is_human_strict"] = g_events["is_human"] & ~trailer_flag
    grp2 = g_events.groupby(["module_id", "month"]).agg(n_total=("is_human_strict", "size"), n_human_strict=("is_human_strict", "sum"))
    grp2["H_G_prime"] = grp2["n_human_strict"] / grp2["n_total"]
    panel = panel.merge(grp2[["H_G_prime"]].reset_index(), on=["module_id", "month"], how="left")
    panel["A_prime"] = 1 - panel["H_G_prime"]

    for metric in ["H_I", "H_G", "H_E"]:
        n_col = f"{metric}_n"
        panel[f"{metric}_low_n"] = panel[n_col].fillna(0) < 3

    processed_d = PROCESSED_DIR / repo_dir_name(repo)
    processed_d.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(processed_d / "panel.parquet", index=False)

    n_modules = panel["module_id"].nunique()
    n_months = panel["month"].nunique()
    total_cells = len(panel)

    def valid_cell_ratio(metric: str) -> float | None:
        n_col = f"{metric}_n"
        valid = (panel[n_col].fillna(0) >= 3).sum()
        return valid / total_cells if total_cells else None

    summary = {
        "repo": repo,
        "n_modules_with_events": n_modules,
        "n_months": n_months,
        "n_panel_rows": total_cells,
        "valid_cell_ratio": {m: valid_cell_ratio(m) for m in ["H_I", "H_G", "H_E"]},
        "A_summary": {
            "mean": float(panel["A"].mean(skipna=True)),
            "std": float(panel["A"].std(skipna=True)),
            "n_non_null": int(panel["A"].notna().sum()),
        },
        "A_prime_summary(trailer込み感度分析)": {
            "mean": float(panel["A_prime"].mean(skipna=True)),
            "std": float(panel["A_prime"].std(skipna=True)),
            "n_non_null": int(panel["A_prime"].notna().sum()),
        },
    }
    (processed_d / "loop4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== {repo} ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"panel.parquet -> {processed_d / 'panel.parquet'}")


if __name__ == "__main__":
    main()

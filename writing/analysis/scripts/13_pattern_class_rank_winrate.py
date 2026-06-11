# -*- coding: utf-8 -*-
"""主張 6: 動的軌跡パターン × IRL/RF rank 勝率 (閾値フリー).

cold IRL は未校正のため、確率の絶対値や閾値 (top 30% 等) には依存しない指標で
IRL と RF の相対競争力を比較する。

評価指標:
  - per-(dir, horizon) cell 内で `pd.Series.rank(ascending=False, pct=True)` を取ると、
    値が高いほど pct が小さい (top 1% → pct=0.01)
  - irl_pct < rf_pct のとき IRL が rank 上位 = IRL win
  - irl_pct > rf_pct のとき RF が rank 上位 = RF win
  - irl_pct == rf_pct のとき tie
  - rank_diff = rf_pct - irl_pct (正 = IRL が上位)

パターン定義:
  pair の 4 区間ラベル (label_0-3m, label_3-6m, label_6-9m, label_9-12m) を
  二値文字列 (例 "1011") にし、以下 7 クラスへマップ:

    A. Steady       1111
    B. Dropout      1000 / 1100 / 1110
    C. Late starter 0001 / 0011 / 0111
    D. Returner     1001 / 1011 / 1101
    E. Sporadic     1010 / 0101
    F. Single-pulse 0010 / 0100 / 0110
    Z. Inactive     0000           (正例なし → 集計対象外)

CI ボット 6 アカウントは除外 (outputs/analysis/bot_detection/bot_candidates_main32.csv)。

入力:
  data/pair_trajectories.csv
  outputs/analysis/bot_detection/bot_candidates_main32.csv (ボット email)

出力:
  results/pattern_class_rank_winrate.csv
  results/fig_pattern_class_rank_winrate.pdf
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
REPO_ROOT = ROOT.parents[1]
BOT_CSV = REPO_ROOT / "outputs/analysis/bot_detection/bot_candidates_main32.csv"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]

PATTERN_TO_CLASS = {
    "1111": ("A", "Steady"),
    "1000": ("B", "Dropout"),
    "1100": ("B", "Dropout"),
    "1110": ("B", "Dropout"),
    "0001": ("C", "Late starter"),
    "0011": ("C", "Late starter"),
    "0111": ("C", "Late starter"),
    "1001": ("D", "Returner"),
    "1011": ("D", "Returner"),
    "1101": ("D", "Returner"),
    "1010": ("E", "Sporadic"),
    "0101": ("E", "Sporadic"),
    "0010": ("F", "Single-pulse"),
    "0100": ("F", "Single-pulse"),
    "0110": ("F", "Single-pulse"),
    "0000": ("Z", "Inactive"),
}
CLASS_ORDER = ["A", "B", "C", "D", "E", "F"]
CLASS_JP = {
    "A": "A. Steady (1111)",
    "B": "B. Dropout (1000/1100/1110)",
    "C": "C. Late starter (0001/0011/0111)",
    "D": "D. Returner (1001/1011/1101)",
    "E": "E. Sporadic (1010/0101)",
    "F": "F. Single-pulse (0010/0100/0110)",
}


def build_pair_pattern(df: pd.DataFrame) -> pd.Series:
    """4 区間ラベルから二値文字列を返す."""
    bits = [df[f"label_{h}"].astype(int).astype(str) for h in HORIZONS]
    pat = bits[0]
    for b in bits[1:]:
        pat = pat + b
    return pat


def percentile_rank_per_cell(df: pd.DataFrame, score_col: str, h: str) -> pd.Series:
    """per-(directory) cell 内で score の percentile rank を取る (high score → low pct)."""
    return df.groupby("directory")[score_col].rank(ascending=False, pct=True, method="average")


def main() -> None:
    pair = pd.read_csv(DATA / "pair_trajectories.csv")
    print(f"raw pairs: {len(pair):,}")

    # ── ボット除外 ────────────────────────────────────
    if BOT_CSV.exists():
        bot_df = pd.read_csv(BOT_CSV)
        bot_emails = set(bot_df.loc[bot_df["is_bot"] == True, "email"].tolist())  # noqa: E712
        n_before = len(pair)
        pair = pair[~pair["developer"].isin(bot_emails)].copy()
        print(f"bot exclusion: {n_before:,} → {len(pair):,}  (-{n_before - len(pair):,})")
    else:
        print(f"[warn] bot list not found at {BOT_CSV} — proceeding without exclusion")

    # ── pair pattern と class ─────────────────────────
    pair["pattern"] = build_pair_pattern(pair)
    pair["class"] = pair["pattern"].map(lambda p: PATTERN_TO_CLASS.get(p, ("?", "?"))[0])
    pair["class_jp"] = pair["pattern"].map(lambda p: PATTERN_TO_CLASS.get(p, ("?", "?"))[1])

    # ── horizon ごとに percentile rank を取り long 形式に展開 ───
    rows = []
    for h in HORIZONS:
        sub = pair.copy()
        sub["ens_h"] = (sub[f"irl_{h}"] + sub[f"rf_{h}"]) / 2.0
        sub["irl_pct"] = percentile_rank_per_cell(sub, f"irl_{h}", h)
        sub["rf_pct"] = percentile_rank_per_cell(sub, f"rf_{h}", h)
        sub["ens_pct"] = percentile_rank_per_cell(sub, "ens_h", h)
        sub_pos = sub[sub[f"label_{h}"] == 1].copy()
        sub_pos["horizon"] = h
        rows.append(sub_pos[["developer", "directory", "horizon", "pattern", "class",
                             "class_jp", "irl_pct", "rf_pct", "ens_pct"]])
    pos = pd.concat(rows, ignore_index=True)
    print(f"positives (label=1, post-bot): {len(pos):,}")

    # ── rank 比較 ─────────────────────────────────────
    pos["irl_wins"] = (pos["irl_pct"] < pos["rf_pct"]).astype(int)
    pos["rf_wins"] = (pos["rf_pct"] < pos["irl_pct"]).astype(int)
    pos["tie"] = (pos["irl_pct"] == pos["rf_pct"]).astype(int)
    pos["rank_diff"] = pos["rf_pct"] - pos["irl_pct"]  # 正 = IRL 上位
    pos["ens_better_irl"] = (pos["ens_pct"] < pos["irl_pct"]).astype(int)
    pos["ens_better_rf"] = (pos["ens_pct"] < pos["rf_pct"]).astype(int)

    # ── パターンクラスごとに集計 ───────────────────────
    out_rows = []
    for cls in CLASS_ORDER + ["ALL"]:
        if cls == "ALL":
            sub = pos[pos["class"].isin(CLASS_ORDER)].copy()
        else:
            sub = pos[pos["class"] == cls].copy()
        n = len(sub)
        if n == 0:
            continue
        irl_w = sub["irl_wins"].sum()
        rf_w = sub["rf_wins"].sum()
        tie = sub["tie"].sum()
        # 二項検定: H0: irl_wins / (irl_wins + rf_wins) = 0.5 (tie 除外)
        n_decisive = irl_w + rf_w
        p_val = stats.binomtest(irl_w, n_decisive, 0.5).pvalue if n_decisive > 0 else np.nan
        out_rows.append({
            "class": cls,
            "class_jp": CLASS_JP.get(cls, "全体 (パターン横断)"),
            "n_pos": n,
            "irl_win_rate": irl_w / n,
            "rf_win_rate": rf_w / n,
            "tie_rate": tie / n,
            "mean_rank_diff": sub["rank_diff"].mean(),
            "median_rank_diff": sub["rank_diff"].median(),
            "ens_better_irl_rate": sub["ens_better_irl"].mean(),
            "ens_better_rf_rate": sub["ens_better_rf"].mean(),
            "p_binom_irl_vs_rf": p_val,
        })
    summary = pd.DataFrame(out_rows)
    print("\n=== パターンクラス × IRL/RF rank 勝率 ===")
    cols = ["class_jp", "n_pos", "irl_win_rate", "rf_win_rate", "tie_rate",
            "mean_rank_diff", "p_binom_irl_vs_rf"]
    print(summary[cols].to_string(index=False))

    out_csv = OUT / "pattern_class_rank_winrate.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")

    # ── 具体例: 各クラスで rank_diff が極端な正例 ─────
    pos["abs_rank_diff"] = pos["rank_diff"].abs()
    example_rows = []
    for cls in ["D", "C", "B"]:  # Returner, Late starter, Dropout
        sub = pos[pos["class"] == cls].copy()
        if cls == "B":
            # Dropout は RF が勝つ tail を見たい → rank_diff 負側の絶対値大
            sub = sub[sub["rank_diff"] < 0].nsmallest(8, "rank_diff")
        else:
            # Returner, Late starter は IRL が勝つ tail → rank_diff 正側
            sub = sub[sub["rank_diff"] > 0].nlargest(8, "rank_diff")
        sub["class_jp"] = CLASS_JP[cls]
        example_rows.append(sub[["class_jp", "developer", "directory", "horizon",
                                  "pattern", "irl_pct", "rf_pct", "rank_diff"]])
    examples = pd.concat(example_rows, ignore_index=True)
    examples_csv = OUT / "pattern_class_rank_top_examples.csv"
    examples.to_csv(examples_csv, index=False)
    print(f"saved: {examples_csv}")
    print("\n=== 各クラスで rank_diff 絶対値が最大の正例 (top 8) ===")
    print(examples.to_string(index=False))

    # ── 図 ─────────────────────────────────────────────
    plot_main_figure(summary, OUT / "fig_pattern_class_rank_winrate.pdf")


def plot_main_figure(summary: pd.DataFrame, out_path: pathlib.Path) -> None:
    cls_rows = summary[summary["class"].isin(CLASS_ORDER)].copy()
    # n_pos の少ない順 = 動的パターン → Steady の順で並べる (主張上の物語順)
    # 表と一致する順序: E, F, C, D, B, A (IRL win rate 降順)
    cls_rows = cls_rows.sort_values("irl_win_rate", ascending=True).reset_index(drop=True)
    all_row = summary[summary["class"] == "ALL"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})

    # ── 左パネル: IRL/RF/tie 勝率の水平積み上げ + ALL ────
    ax = axes[0]
    y = np.arange(len(cls_rows) + 1)  # +1 for ALL
    labels = cls_rows["class_jp"].tolist() + ["全体 (パターン横断)"]
    irl = np.append(cls_rows["irl_win_rate"].values, all_row["irl_win_rate"])
    rf = np.append(cls_rows["rf_win_rate"].values, all_row["rf_win_rate"])
    tie = np.append(cls_rows["tie_rate"].values, all_row["tie_rate"])
    n_pos = np.append(cls_rows["n_pos"].values, all_row["n_pos"])
    p_vals = np.append(cls_rows["p_binom_irl_vs_rf"].values, all_row["p_binom_irl_vs_rf"])

    ax.barh(y, irl, color="#1f77b4", alpha=0.88, label="IRL rank 上位")
    ax.barh(y, rf, left=irl, color="#d62728", alpha=0.88, label="RF rank 上位")
    ax.barh(y, tie, left=irl + rf, color="#bbbbbb", alpha=0.7, label="tie")
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("正例 1 件あたりの rank 比較結果の割合")
    ax.set_title("(a) IRL/RF rank 勝率 (pattern クラス別、閾値フリー)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    # 注釈: IRL win rate と n_pos
    for i, (rate, n, p) in enumerate(zip(irl, n_pos, p_vals)):
        sig = ""
        if not np.isnan(p):
            if p < 0.001:
                sig = " ★"
            elif p < 0.01:
                sig = " **"
            elif p < 0.05:
                sig = " *"
        ax.text(rate / 2, i, f"{rate:.3f}{sig}",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(1.005, i, f"n={int(n):,}", ha="left", va="center", fontsize=8.5, color="#333")
    # ALL の行をハイライト
    ax.axhline(len(cls_rows) - 0.5, color="black", linestyle="-", linewidth=0.6, alpha=0.4)

    # ── 右パネル: mean rank_diff ────────────────────────
    ax = axes[1]
    mean_rd = np.append(cls_rows["mean_rank_diff"].values, all_row["mean_rank_diff"])
    colors = ["#1f77b4" if v > 0 else "#d62728" for v in mean_rd]
    ax.barh(y, mean_rd, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("mean rank_diff = mean(rf_pct − irl_pct)\n正 = IRL が平均的に rank 上位")
    ax.set_title("(b) 平均 rank 差 (符号 = どちらが上位)", fontsize=11)
    # 数値注釈
    for i, v in enumerate(mean_rd):
        ha = "left" if v >= 0 else "right"
        offset = 0.001 if v >= 0 else -0.001
        ax.text(v + offset, i, f"{v:+.3f}", ha=ha, va="center", fontsize=9)
    rng = max(abs(mean_rd.min()), abs(mean_rd.max())) * 1.4
    ax.set_xlim(-rng, rng)
    ax.axhline(len(cls_rows) - 0.5, color="black", linestyle="-", linewidth=0.6, alpha=0.4)

    fig.suptitle("動的軌跡パターン × IRL/RF rank 競争力 (cold IRL、CI ボット除外後 n="
                 f"{int(all_row['n_pos']):,} 正例)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

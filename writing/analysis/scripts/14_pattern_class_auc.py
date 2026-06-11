# -*- coding: utf-8 -*-
"""主張 6 補強: 動的軌跡パターン × クラス内 AUC (絶対的予測性能の比較).

主張 6 の rank 勝率 (`13_pattern_class_rank_winrate.py`) は IRL/RF の相対的な
rank 上下のみを見ていたため、「どちらも正例を当ててもどちらか勝つ」「どちらも
外しても勝負がつく」という limitation があった。本スクリプトはこれを補完するため、
パターンクラスごとの **絶対 AUC** を計算する。

評価方法:
  各パターンクラス C について、
    - 母集団 = pair の pattern が C に属する全 (pair, horizon) 行
    - score = irl_h / rf_h / avg_h (= Ens)
    - label = label_h ∈ {0, 1}
  この母集団で標準 ROC AUC を計算する。

注意:
  - **Steady (1111)** は全 label=1 のため AUC 計算不能 → 集計対象外
  - **Inactive (0000)** は全 label=0 のため AUC 計算不能 → 集計対象外
  - 計算可能なクラス: B. Dropout / C. Late starter / D. Returner / E. Sporadic / F. Single-pulse
  - 全体 (パターン横断) も参照値として計算 (Steady/Inactive 含む全 (pair, horizon))

入力:
  data/pair_trajectories.csv
  outputs/analysis/bot_detection/bot_candidates_main32.csv

出力:
  results/pattern_class_auc.csv
  results/fig_pattern_class_auc.pdf
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
REPO_ROOT = ROOT.parents[1]
BOT_CSV = REPO_ROOT / "outputs/analysis/bot_detection/bot_candidates_main32.csv"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]

PATTERN_TO_CLASS = {
    "1111": ("A", "A. Steady (1111)"),
    "1000": ("B", "B. Dropout (1000/1100/1110)"),
    "1100": ("B", "B. Dropout (1000/1100/1110)"),
    "1110": ("B", "B. Dropout (1000/1100/1110)"),
    "0001": ("C", "C. Late starter (0001/0011/0111)"),
    "0011": ("C", "C. Late starter (0001/0011/0111)"),
    "0111": ("C", "C. Late starter (0001/0011/0111)"),
    "1001": ("D", "D. Returner (1001/1011/1101)"),
    "1011": ("D", "D. Returner (1001/1011/1101)"),
    "1101": ("D", "D. Returner (1001/1011/1101)"),
    "1010": ("E", "E. Sporadic (1010/0101)"),
    "0101": ("E", "E. Sporadic (1010/0101)"),
    "0010": ("F", "F. Single-pulse (0010/0100/0110)"),
    "0100": ("F", "F. Single-pulse (0010/0100/0110)"),
    "0110": ("F", "F. Single-pulse (0010/0100/0110)"),
    "0000": ("Z", "Z. Inactive (0000)"),
}
COMPUTABLE_CLASSES = ["B", "C", "D", "E", "F"]


def build_pattern(df: pd.DataFrame) -> pd.Series:
    bits = [df[f"label_{h}"].astype(int).astype(str) for h in HORIZONS]
    pat = bits[0]
    for b in bits[1:]:
        pat = pat + b
    return pat


def safe_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def to_long(pair: pd.DataFrame) -> pd.DataFrame:
    """wide → long: 1 行 / (developer, directory, horizon)."""
    rows = []
    for h in HORIZONS:
        sub = pair[["developer", "directory", "pattern", "class", "class_jp",
                    f"label_{h}", f"irl_{h}", f"rf_{h}"]].copy()
        sub.columns = ["developer", "directory", "pattern", "class", "class_jp",
                       "label", "irl", "rf"]
        sub["horizon"] = h
        sub["ens"] = (sub["irl"] + sub["rf"]) / 2.0
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    pair = pd.read_csv(DATA / "pair_trajectories.csv")
    print(f"raw pairs: {len(pair):,}")

    # ── bot 除外 ───────────────────────────────────────
    if BOT_CSV.exists():
        bot_df = pd.read_csv(BOT_CSV)
        bots = set(bot_df.loc[bot_df["is_bot"] == True, "email"].tolist())  # noqa: E712
        n0 = len(pair)
        pair = pair[~pair["developer"].isin(bots)].copy()
        print(f"bot exclusion: {n0:,} → {len(pair):,}  (-{n0 - len(pair):,})")

    pair["pattern"] = build_pattern(pair)
    pair["class"] = pair["pattern"].map(lambda p: PATTERN_TO_CLASS.get(p, ("?", "?"))[0])
    pair["class_jp"] = pair["pattern"].map(lambda p: PATTERN_TO_CLASS.get(p, ("?", "?"))[1])

    long = to_long(pair)
    print(f"long rows (pair × 4 horizons): {len(long):,}")

    # ── AUC 集計 ─────────────────────────────────────
    # (1) pooled AUC: 全 (pair, horizon) を pool して 1 つの AUC を計算
    #     → 「ペア間で活動的な dev を識別」+「ペア内で active 月を識別」の混合
    # (2) within-pair AUC: 1 ペアごとに pos vs neg horizon を比較し、ペア平均
    #     → 純粋な「ペア内 temporal localization」能力
    # 両者の対比で「pooled AUC の高さは temporal localization 由来か、
    # ペア間判別由来か」を切り分けられる。

    def within_pair_auc(pair_subdf: pd.DataFrame, score_col: str) -> tuple[float, int]:
        """各 (developer, directory) ペアの 4 horizon で AUC を出し、計算可能なペアの単純平均."""
        aucs = []
        for (_dev, _dir), g in pair_subdf.groupby(["developer", "directory"]):
            y = g["label"].astype(int).values
            if len(np.unique(y)) < 2:
                continue
            aucs.append(safe_auc(y, g[score_col].values))
        if not aucs:
            return np.nan, 0
        return float(np.mean(aucs)), len(aucs)

    rows = []
    for cls in COMPUTABLE_CLASSES + ["ALL_mixed", "ALL_with_steady_inactive"]:
        if cls == "ALL_mixed":
            sub = long[long["class"].isin(COMPUTABLE_CLASSES)]
            label_jp = "全体 (混合パターンのみ)"
        elif cls == "ALL_with_steady_inactive":
            sub = long
            label_jp = "全体 (Steady/Inactive 含む参照値)"
        else:
            sub = long[long["class"] == cls]
            label_jp = sub["class_jp"].iloc[0]
        n = len(sub)
        n_pos = int(sub["label"].sum())
        base_rate = n_pos / n if n > 0 else np.nan
        # within-pair AUC
        irl_wp, n_pairs_used = within_pair_auc(sub, "irl")
        rf_wp, _ = within_pair_auc(sub, "rf")
        ens_wp, _ = within_pair_auc(sub, "ens")
        rows.append({
            "class": cls,
            "class_jp": label_jp,
            "n_rows": n,
            "n_pos": n_pos,
            "base_rate": base_rate,
            "n_pairs_within_auc": n_pairs_used,
            "irl_auc_pooled": safe_auc(sub["label"], sub["irl"]),
            "rf_auc_pooled": safe_auc(sub["label"], sub["rf"]),
            "ens_auc_pooled": safe_auc(sub["label"], sub["ens"]),
            "irl_auc_within_pair": irl_wp,
            "rf_auc_within_pair": rf_wp,
            "ens_auc_within_pair": ens_wp,
        })
    summary = pd.DataFrame(rows)
    summary["delta_irl_minus_rf_pooled"] = summary["irl_auc_pooled"] - summary["rf_auc_pooled"]
    summary["delta_irl_minus_rf_within"] = summary["irl_auc_within_pair"] - summary["rf_auc_within_pair"]

    print("\n=== パターンクラス × AUC (pooled vs within-pair) ===")
    cols = ["class_jp", "n_rows", "n_pos", "n_pairs_within_auc",
            "irl_auc_pooled", "rf_auc_pooled", "ens_auc_pooled",
            "irl_auc_within_pair", "rf_auc_within_pair", "ens_auc_within_pair",
            "delta_irl_minus_rf_pooled", "delta_irl_minus_rf_within"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out_csv = OUT / "pattern_class_auc.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")

    # ── 図 ─────────────────────────────────────────────
    plot_main_figure(summary, OUT / "fig_pattern_class_auc.pdf")


def plot_main_figure(summary: pd.DataFrame, out_path: pathlib.Path) -> None:
    sub = summary[summary["class"].isin(COMPUTABLE_CLASSES)].copy()
    sub = sub.sort_values("irl_auc_pooled", ascending=True).reset_index(drop=True)
    all_row = summary[summary["class"] == "ALL_mixed"].iloc[0]

    labels = sub["class_jp"].tolist() + ["全体 (混合のみ、参照)"]
    y = np.arange(len(labels))

    def col_with_all(field: str) -> np.ndarray:
        return np.append(sub[field].values, all_row[field])

    irl_p = col_with_all("irl_auc_pooled")
    rf_p = col_with_all("rf_auc_pooled")
    ens_p = col_with_all("ens_auc_pooled")
    irl_w = col_with_all("irl_auc_within_pair")
    rf_w = col_with_all("rf_auc_within_pair")
    ens_w = col_with_all("ens_auc_within_pair")
    n_rows = col_with_all("n_rows")
    n_pairs = col_with_all("n_pairs_within_auc")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    # ── 左パネル: pooled AUC (ペア間+ペア内 混合) ────
    ax = axes[0]
    hbar = 0.27
    ax.barh(y + hbar, irl_p, height=hbar, color="#1f77b4", alpha=0.88, label="IRL")
    ax.barh(y, rf_p, height=hbar, color="#d62728", alpha=0.88, label="RF")
    ax.barh(y - hbar, ens_p, height=hbar, color="#2ca02c", alpha=0.88, label="Ens")
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0.3, 0.8)
    ax.set_xlabel("Pooled AUC (全 (pair, horizon) 行を pool)")
    ax.set_title("(a) Pooled AUC = ペア間判別 + ペア内 temporal の混合", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    for i in range(len(labels)):
        ax.text(irl_p[i] + 0.003, i + hbar, f"{irl_p[i]:.3f}",
                va="center", fontsize=8.5, color="#1f77b4")
        ax.text(rf_p[i] + 0.003, i, f"{rf_p[i]:.3f}",
                va="center", fontsize=8.5, color="#d62728")
        ax.text(ens_p[i] + 0.003, i - hbar, f"{ens_p[i]:.3f}",
                va="center", fontsize=8.5, color="#2ca02c")
        ax.text(0.305, i + hbar, f"n_rows={int(n_rows[i]):,}",
                va="center", fontsize=7.5, color="#666")
    ax.axhline(len(sub) - 0.5, color="black", linestyle="-", linewidth=0.6, alpha=0.4)

    # ── 右パネル: within-pair AUC (純粋 temporal) ───
    ax = axes[1]
    ax.barh(y + hbar, irl_w, height=hbar, color="#1f77b4", alpha=0.88, label="IRL")
    ax.barh(y, rf_w, height=hbar, color="#d62728", alpha=0.88, label="RF")
    ax.barh(y - hbar, ens_w, height=hbar, color="#2ca02c", alpha=0.88, label="Ens")
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0.3, 0.8)
    ax.set_xlabel("Within-pair AUC (1 ペア内で active vs inactive 月を識別)")
    ax.set_title("(b) Within-pair AUC = 純粋な temporal localization", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    for i in range(len(labels)):
        ax.text(irl_w[i] + 0.003, i + hbar, f"{irl_w[i]:.3f}",
                va="center", fontsize=8.5, color="#1f77b4")
        ax.text(rf_w[i] + 0.003, i, f"{rf_w[i]:.3f}",
                va="center", fontsize=8.5, color="#d62728")
        ax.text(ens_w[i] + 0.003, i - hbar, f"{ens_w[i]:.3f}",
                va="center", fontsize=8.5, color="#2ca02c")
        ax.text(0.305, i + hbar, f"n_pairs={int(n_pairs[i]):,}",
                va="center", fontsize=7.5, color="#666")
    ax.axhline(len(sub) - 0.5, color="black", linestyle="-", linewidth=0.6, alpha=0.4)

    fig.suptitle(
        "動的軌跡パターン × AUC 比較: (a) pooled (混合)  vs  (b) within-pair (純粋 temporal)",
        fontsize=12, y=1.00
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

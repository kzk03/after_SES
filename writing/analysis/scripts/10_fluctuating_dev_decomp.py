"""
§11: 揺らぎ開発者の分解 (同一ペア揺れ vs 承諾相手の入れ替わり)

§4b で確認したとおり、開発者単位で 4 予測区間のラベルが揺れる開発者は 152 名 (35%) いる。
教授の問い (2)「予測区間で予測しているから周期的な揺らぎが効くか」の真相を、揺れの起源で分解する:

  - 同一ペア揺れ型: 同一の (開発者, ディレクトリ) ペアのラベルが予測区間を跨いで揺れる
    (= 同じ相手に対して状態が変化)
  - 承諾相手の入れ替わり型: 異なるディレクトリのペアが予測区間ごとに承諾の主役を入れ替える
    (= 開発者の活動先 (どのディレクトリでレビュー受理するか) が時系列で変わる)

分解ロジック (各揺らぎ開発者について):
  各ペアを 3 種類に分類:
    - 切替    : ラベルが 4 予測区間中で 0 と 1 の両方を持つ (ペア単位で揺らぐ)
    - 全承諾 : 4 予測区間すべてでラベル=1
    - 全拒否 : 4 予測区間すべてでラベル=0
  そして開発者を次の 3 タイプに分類:
    - 切替主導型      : 切替ペアが 1 個以上 (= 同一ペアの揺れが寄与している)
    - 承諾先入替純粋型 : 切替ペアが 0 個だが開発者単位でラベルが揺れる
                         (= 異なる全承諾ペアが予測区間ごとに交代で承諾する)
                         → ただしペア単位の max 集約では全予測区間で 1 のはず。揺れるのは
                         　全承諾も全拒否もなく「開発者は複数ディレクトリを持ち、予測区間ごとに別のディレクトリで承諾」
    - 混合型          : 切替ペアあり + 全承諾ペアあり (両方の機序)

出力:
  - results/fluctuating_dev_decomp.csv: 開発者単位の決定
  - results/fluctuating_dev_summary.json
  - results/fig_fluctuating_dev_decomp.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def classify_pair(labels: np.ndarray) -> str:
    """ペア単位の label シーケンスを 3 種類に分類."""
    has_pos = (labels == 1).any()
    has_neg = (labels == 0).any()
    if has_pos and has_neg:
        return "flip"
    if has_pos:
        return "all_acc"
    return "all_rej"


def main() -> None:
    pair = pd.read_csv(DATA / "pair_trajectories.csv")
    label_cols = [f"label_{h}" for h in HORIZONS]

    # pair classification
    pair["pair_class"] = pair[label_cols].apply(
        lambda row: classify_pair(row.values), axis=1
    )

    # dev レベル max-aggregated label (any-positive)
    dev_max = pair.groupby("developer")[label_cols].max().reset_index()
    dev_max["n_horizons_pos"] = dev_max[label_cols].sum(axis=1)
    dev_max["dev_label_class"] = dev_max.apply(
        lambda row: (
            "consistent_reject" if row["n_horizons_pos"] == 0 else
            "consistent_accept" if row["n_horizons_pos"] == 4 else
            "fluctuating"
        ), axis=1
    )

    # dev 単位のペア構成
    pair_summary = pair.groupby(["developer", "pair_class"]).size().unstack(fill_value=0)
    for c in ["flip", "all_acc", "all_rej"]:
        if c not in pair_summary.columns:
            pair_summary[c] = 0
    pair_summary = pair_summary[["flip", "all_acc", "all_rej"]].reset_index()
    pair_summary["n_pairs_total"] = pair_summary[["flip", "all_acc", "all_rej"]].sum(axis=1)

    dev = dev_max.merge(pair_summary, on="developer", how="left")

    # type 決定
    def decompose(row: pd.Series) -> str:
        if row["dev_label_class"] != "fluctuating":
            return row["dev_label_class"]
        n_flip = row["flip"]
        n_acc = row["all_acc"]
        if n_flip == 0:
            # 揺れているのにペアが flip しない = 異なる all_acc ペアが horizon を交代
            return "portfolio_shift_pure"
        if n_acc == 0:
            return "biorhythm_dominant"
        return "mixed"

    dev["decomp_type"] = dev.apply(decompose, axis=1)

    # ---- Jaccard 分析: fluctuating dev で、accept する horizon 間で
    # 「同じペアが accept しているか」 vs 「異なるペアが交代しているか」 ----
    jaccard_rows = []
    fluct_devs = set(dev[dev["dev_label_class"] == "fluctuating"]["developer"])
    pair_idx = pair.set_index(["developer", "directory"])
    for d, grp in pair.groupby("developer"):
        if d not in fluct_devs:
            continue
        accepting_sets = []
        for h in HORIZONS:
            accs = set(grp.loc[grp[f"label_{h}"] == 1, "directory"].tolist())
            if accs:
                accepting_sets.append(accs)
        if len(accepting_sets) < 2:
            continue
        shared = set.intersection(*accepting_sets)
        union = set.union(*accepting_sets)
        ratio = len(shared) / len(union) if union else 0.0
        jaccard_rows.append({
            "developer": d,
            "n_accept_horizons": len(accepting_sets),
            "n_shared": len(shared),
            "n_union": len(union),
            "jaccard": ratio,
        })
    jac_df = pd.DataFrame(jaccard_rows)
    if not jac_df.empty:
        # categorize: high (>=0.5) = same pair drives, low (<0.5) = rotation
        jac_df["rotation_class"] = np.where(jac_df["jaccard"] >= 0.5, "same_pair_drives", "rotation")
        jac_df.to_csv(RESULTS / "fluctuating_dev_jaccard.csv", index=False)
        dev = dev.merge(jac_df[["developer", "jaccard", "rotation_class"]], on="developer", how="left")

    dev.to_csv(RESULTS / "fluctuating_dev_decomp.csv", index=False)

    # summary
    fluct = dev[dev["dev_label_class"] == "fluctuating"]
    counts = fluct["decomp_type"].value_counts().to_dict()
    n_fluct = int(len(fluct))
    summary = {
        "n_devs_total": int(len(dev)),
        "n_devs_fluctuating": n_fluct,
        "decomp_counts": {k: int(v) for k, v in counts.items()},
        "decomp_ratios": {k: float(v) / n_fluct for k, v in counts.items()},
    }

    # additional: among fluctuating devs, distribution of flip-pair ratio
    fluct = fluct.copy()
    fluct["flip_ratio"] = fluct["flip"] / fluct["n_pairs_total"]
    summary["flip_ratio_quartiles"] = {
        "p25": float(fluct["flip_ratio"].quantile(0.25)),
        "p50": float(fluct["flip_ratio"].quantile(0.50)),
        "p75": float(fluct["flip_ratio"].quantile(0.75)),
        "mean": float(fluct["flip_ratio"].mean()),
    }

    # how many fluctuating devs have at least 1 flip pair?
    summary["n_fluct_with_flip_pair"] = int((fluct["flip"] >= 1).sum())
    summary["n_fluct_zero_flip_pair"] = int((fluct["flip"] == 0).sum())

    # jaccard rotation summary (already merged into dev, re-pull from there)
    if "jaccard" in dev.columns:
        fluct_with_jac = dev[(dev["dev_label_class"] == "fluctuating") & dev["jaccard"].notna()]
        if len(fluct_with_jac) > 0:
            summary["jaccard"] = {
                "n_with_jaccard": int(len(fluct_with_jac)),
                "mean": float(fluct_with_jac["jaccard"].mean()),
                "median": float(fluct_with_jac["jaccard"].median()),
                "n_rotation_jac_lt_0p5": int((fluct_with_jac["jaccard"] < 0.5).sum()),
                "n_same_pair_jac_ge_0p5": int((fluct_with_jac["jaccard"] >= 0.5).sum()),
            }

    (RESULTS / "fluctuating_dev_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- additional: AUC by decomp type (re-check biorhythm hypothesis with sharper lens) ----
    from sklearn.metrics import roc_auc_score

    irl_cols = [f"irl_{h}" for h in HORIZONS]
    rf_cols = [f"rf_{h}" for h in HORIZONS]
    avg_cols = [f"avg_{h}" for h in HORIZONS]
    pair_with_type = pair.merge(dev[["developer", "decomp_type"]], on="developer", how="left")

    auc_rows = []
    for dtype, grp in pair_with_type.groupby("decomp_type"):
        for h in HORIZONS:
            y = grp[f"label_{h}"].values
            if len(np.unique(y)) < 2:
                continue
            irl = grp[f"irl_{h}"].values
            rf = grp[f"rf_{h}"].values
            avg = grp[f"avg_{h}"].values
            auc_rows.append({
                "decomp_type": dtype,
                "horizon": h,
                "n_pairs": int(len(grp)),
                "pos_rate": float(y.mean()),
                "auc_irl": float(roc_auc_score(y, irl)),
                "auc_rf": float(roc_auc_score(y, rf)),
                "auc_ens": float(roc_auc_score(y, avg)),
            })
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(RESULTS / "fluctuating_dev_auc_by_type.csv", index=False)
    print("\nAUC by decomp type:")
    print(auc_df.to_string(index=False))

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # left: pie chart of Jaccard-based decomposition (相手入れ替わり vs 同一ペア揺れ)
    ax = axes[0]
    if "jaccard" in dev.columns:
        fluct_jac = dev[(dev["dev_label_class"] == "fluctuating") & dev["jaccard"].notna()]
        n_rot = int((fluct_jac["jaccard"] < 0.5).sum())
        n_same = int((fluct_jac["jaccard"] >= 0.5).sum())
        n_single = int(((dev["dev_label_class"] == "fluctuating") & dev["jaccard"].isna()).sum())
        sizes = [n_rot, n_same, n_single]
        labels = [
            f"承諾相手の入れ替わり\n(Jaccard<0.5)\nn={n_rot}",
            f"同一ペア揺れ\n(Jaccard>=0.5)\nn={n_same}",
            f"単一承諾区間\n(分解不可)\nn={n_single}",
        ]
        colors = ["#ff7f0e", "#1f77b4", "#cccccc"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 9})
        ax.set_title(f"揺らぎ開発者の Jaccard 分解 (n={n_fluct})")
    else:
        ax.text(0.5, 0.5, "Jaccard 算出不能", ha="center", va="center")

    # middle: flip_ratio histogram
    ax = axes[1]
    ax.hist(fluct["flip_ratio"], bins=20, color="#2ca02c", alpha=0.75, edgecolor="black")
    ax.axvline(fluct["flip_ratio"].mean(), color="red", linestyle="--",
               label=f"mean={fluct['flip_ratio'].mean():.2f}")
    ax.axvline(fluct["flip_ratio"].median(), color="orange", linestyle="--",
               label=f"median={fluct['flip_ratio'].median():.2f}")
    ax.set_xlabel("切替ペア比率 (揺らぎ開発者ごと)")
    ax.set_ylabel("開発者数")
    ax.set_title("揺らぎ開発者内の切替ペア比率分布")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # right: AUC by decomp type (averaged across horizons)
    ax = axes[2]
    auc_pivot = (auc_df.groupby("decomp_type")[["auc_irl", "auc_rf", "auc_ens"]]
                       .mean()
                       .reindex(["biorhythm_dominant", "mixed", "portfolio_shift_pure",
                                 "consistent_accept", "consistent_reject"])
                       .dropna(how="all"))
    x = np.arange(len(auc_pivot))
    w = 0.27
    ax.bar(x - w, auc_pivot["auc_irl"], width=w, label="IRL", color="#1f77b4")
    ax.bar(x, auc_pivot["auc_rf"], width=w, label="RF", color="#d62728")
    ax.bar(x + w, auc_pivot["auc_ens"], width=w, label="Ens", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels([{
        "biorhythm_dominant": "切替主導型",
        "mixed": "混合型",
        "portfolio_shift_pure": "相手入替\n純粋型",
        "consistent_accept": "常時承諾型",
        "consistent_reject": "常時拒否型"
    }[c] for c in auc_pivot.index], rotation=0, fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, label="ランダム水準")
    ax.set_ylabel("AUC (4 予測区間平均)")
    ax.set_title("分解タイプ別 AUC")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_pdf = RESULTS / "fig_fluctuating_dev_decomp.pdf"
    out_png = RESULTS / "fig_fluctuating_dev_decomp.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n-> {out_pdf}")
    print(f"-> {out_png}")


if __name__ == "__main__":
    main()

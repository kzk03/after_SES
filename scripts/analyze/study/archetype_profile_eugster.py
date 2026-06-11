# -*- coding: utf-8 -*-
"""Eugster (2012) "Performance Profiles based on Archetypal Athletes" 風の archetype 可視化.

NBA 選手の archetype 分析で用いられた可視化スタイルを開発者 archetype に適用:
  - Panel A: 全 archetype overlay radar chart (一目で archetype の "形" 比較)
  - Panel B: archetype × feature の horizontal bar chart matrix (詳細比較)
  - Panel C: discriminative score 順に並べた parallel coordinates
  - Panel D: 各 archetype の "signature feature" top-5 (差分が大きい特徴量)

入力: k{K}_archetype_profile.csv (rows=archetype, cols=feature, values=normalized 0-1)
"""
from __future__ import annotations

import argparse
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 4-archetype labels (B-28 で確定したラベル)
# 表6 (tab:archetypes) の「名前」列と一致させる
ARCH_NAMES = {
    "A1": "低活動型",
    "A2": "短期・レビュー中心型",
    "A3": "高承諾・高活動型",
    "A4": "常時活動・低承諾型",
}
# 表4 (tab:pair-arch) の「名前」列と一致させる
PAIR_NAMES = {
    "P1": "他プロジェクト経験者",
    "P2": "関与低下型",
    "P3": "高承諾型",
    "P4": "低活動・低承諾型",
}
PAIR_COLORS = {
    "P1": "#E64B35",
    "P2": "#4DBBD5",
    "P3": "#00A087",
    "P4": "#F39B7F",
}
# Eugster paper 風カラーパレット (NBA position colors からインスパイア)
ARCH_COLORS = {
    "A1": "#E64B35",  # red — 新人
    "A2": "#4DBBD5",  # blue — レビュー専
    "A3": "#00A087",  # green — 承認ベテラン
    "A4": "#F39B7F",  # orange — 飽和
}
# 特徴量を意味カテゴリでグループ化 (radar/bar の並び順)
FEATURE_GROUPS = [
    ("在籍・量", [
        "window_tenure_days", "total_changes", "total_reviews",
        "active_months_ratio", "n_projects",
    ]),
    ("直近活動", [
        "recent_activity_frequency", "avg_activity_gap",
        "days_since_last_activity", "recent_load_ratio_30d_all",
        "recent_load_ratio_7d_30d", "concurrent_open_reviews_7d",
        "activity_trend",
    ]),
    ("承諾傾向", [
        "overall_acceptance_rate", "recent_acceptance_rate",
        "acceptance_rate_last10", "accept_rate_drift_30d",
        "time_since_last_accept", "time_since_last_reject",
        "recent_switch_rate",
    ]),
    ("役割・ネットワーク", [
        "reviewer_owner_ratio", "core_reviewer_ratio",
        "unique_collaborator_count", "reciprocity_score",
        "cross_project_review_share", "same_domain_share",
        "response_time_trend",
    ]),
]


def shorten(name: str) -> str:
    mapping = {
        "window_tenure_days": "tenure_days",
        "recent_activity_frequency": "act_freq",
        "recent_acceptance_rate": "accept_rate_rec",
        "overall_acceptance_rate": "accept_rate_all",
        "acceptance_rate_last10": "accept_last10",
        "recent_load_ratio_30d_all": "load_30d/all",
        "recent_load_ratio_7d_30d": "load_7d/30d",
        "days_since_last_activity": "days_inactive",
        "unique_collaborator_count": "n_collab",
        "concurrent_open_reviews_7d": "open_reviews_7d",
        "cross_project_review_share": "cross_proj",
        "reciprocity_score": "reciprocity",
        "reviewer_owner_ratio": "rev/owner",
        "core_reviewer_ratio": "core_rev",
        "time_since_last_accept": "tsl_accept",
        "time_since_last_reject": "tsl_reject",
        "accept_rate_drift_30d": "accept_drift",
        "recent_switch_rate": "switch_rate",
        "avg_activity_gap": "act_gap",
        "activity_trend": "act_trend",
        "active_months_ratio": "active_mo",
        "response_time_trend": "resp_trend",
        "same_domain_share": "same_domain",
        "n_projects": "n_proj",
        "total_changes": "n_changes",
        "total_reviews": "n_reviews",
        "path_review_count": "path_n_rev",
        "path_recency": "path_recency",
        "path_acceptance_rate": "path_acc_rate",
        "path_owner_overlap": "path_owner",
        "path_hub_score": "path_hub",
        "path_neighbor_coverage": "path_neigh",
    }
    return mapping.get(name, name)


def panel_overlay_radar(ax, df: pd.DataFrame, ordered_feats: list[str]) -> None:
    N = len(ordered_feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    for a in df.index:
        vals = df.loc[a, ordered_feats].to_numpy()
        vals_closed = np.concatenate([vals, vals[:1]])
        ax.plot(angles_closed, vals_closed, color=ARCH_COLORS[a],
                label=f"{a}: {ARCH_NAMES[a]}", lw=2.0)
        ax.fill(angles_closed, vals_closed, color=ARCH_COLORS[a], alpha=0.15)
    ax.set_xticks(angles)
    ax.set_xticklabels([shorten(f) for f in ordered_feats], fontsize=7)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("(A) Archetype profile radar (all 26 features, min-max normalized)",
                 fontsize=10, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8,
              frameon=False)


def panel_grouped_bars(ax, df: pd.DataFrame) -> None:
    """Eugster NBA paper 風: archetype × feature の grouped horizontal bar."""
    archetypes = list(df.index)
    n_arch = len(archetypes)
    bar_h = 0.8 / n_arch
    ypos = 0
    yticks, ylabels = [], []
    group_separators = []

    for gname, feats in FEATURE_GROUPS:
        feats = [f for f in feats if f in df.columns]
        group_top = ypos
        for f in feats:
            for i, a in enumerate(archetypes):
                v = df.loc[a, f]
                ax.barh(ypos + i * bar_h - bar_h * (n_arch - 1) / 2,
                        v, height=bar_h * 0.9,
                        color=ARCH_COLORS[a], alpha=0.85,
                        edgecolor="white", lw=0.5)
            yticks.append(ypos)
            ylabels.append(shorten(f))
            ypos += 1.2
        # group label
        ax.text(-0.04, (group_top + ypos - 1.2) / 2,
                gname, fontsize=9, rotation=90,
                ha="center", va="center", fontweight="bold",
                transform=ax.get_yaxis_transform())
        group_separators.append(ypos - 0.6)
        ypos += 0.6  # gap between groups

    for sep in group_separators[:-1]:
        ax.axhline(sep, color="lightgray", lw=0.5, zorder=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Normalized feature value", fontsize=8)
    ax.set_title("(B) Archetype × feature profile (grouped bars)",
                 fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ARCH_COLORS[a])
               for a in archetypes]
    labels = [f"{a}: {ARCH_NAMES[a]}" for a in archetypes]
    ax.legend(handles, labels, loc="lower right", fontsize=7, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)


def panel_signature(ax, df: pd.DataFrame, topn: int = 5,
                    standalone: bool = False, pair_mode: bool = False) -> None:
    """各 archetype で他 archetype 平均と最も乖離する feature top-n.

    pair_mode=True のときは行ラベル A1..A4 を P1..P4 として描画し，
    PAIR_NAMES / PAIR_COLORS を用いる．
    """
    if pair_mode:
        label_map = {f"A{i}": f"P{i}" for i in range(1, 5)}
        names = PAIR_NAMES
        colors = PAIR_COLORS
    else:
        label_map = {a: a for a in df.index}
        names = ARCH_NAMES
        colors = ARCH_COLORS

    sig_rows = []
    for a in df.index:
        others = df.drop(index=a).mean(axis=0)
        diff = df.loc[a] - others
        top = diff.abs().nlargest(topn).index
        for f in top:
            sig_rows.append({
                "archetype": label_map[a], "feature": shorten(f),
                "diff": diff[f], "value": df.loc[a, f],
            })
    sig = pd.DataFrame(sig_rows)

    archetypes = [label_map[a] for a in df.index]
    y = 0
    yticks, ylabels = [], []
    for a in archetypes:
        sub = sig[sig["archetype"] == a].sort_values("diff", ascending=False)
        for _, r in sub.iterrows():
            ax.barh(y, r["diff"], color=colors[a],
                    alpha=0.85, edgecolor="white", lw=0.5)
            yticks.append(y)
            ylabels.append(f"{r['feature']}")
            y += 1
        y += 0.6  # gap
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(yticks)
    label_fs = 26 if standalone else 7
    xlabel_fs = 24 if standalone else 8
    title_fs = 28 if standalone else 10
    arch_fs = 26 if standalone else 8
    label_w = "bold" if standalone else "normal"
    ax.set_yticklabels(ylabels, fontsize=label_fs, fontweight=label_w)
    ax.tick_params(axis="x", labelsize=label_fs)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight(label_w)
    ax.invert_yaxis()
    ax.set_xlabel("特徴量値 − 他archetypeの平均", fontsize=xlabel_fs,
                  fontweight=label_w)
    title = (f"archetype別の特徴的な特徴量（上位{topn}）"
             if standalone
             else f"(C) archetype別の特徴的な特徴量（上位{topn}）")
    ax.set_title(title, fontsize=title_fs, fontweight=label_w)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    # archetype labels on right
    cur_y = 0
    for a in archetypes:
        n = topn
        ax.text(1.02, cur_y + (n - 1) / 2,
                f"{a}\n{names[a]}",
                transform=ax.get_yaxis_transform(),
                fontsize=arch_fs, color=colors[a], fontweight="bold",
                va="center", ha="left")
        cur_y += n + 0.6


def panel_alpha_stack(ax, alpha_csv: pathlib.Path) -> None:
    """全 dev の α を dominant archetype で並べた stacked bar."""
    alpha = pd.read_csv(alpha_csv, index_col=0)
    arch_cols = [c for c in ["A1", "A2", "A3", "A4"] if c in alpha.columns]
    a = alpha[arch_cols].copy()
    a["dominant"] = a.idxmax(axis=1)
    # sort: dominant order, then secondary by α value
    a["max_val"] = a[arch_cols].max(axis=1)
    a = a.sort_values(
        ["dominant", "max_val"],
        ascending=[True, False],
        key=lambda x: x.map({"A1": 0, "A2": 1, "A3": 2, "A4": 3})
        if x.name == "dominant" else x,
    )
    x = np.arange(len(a))
    bottom = np.zeros(len(a))
    for col in arch_cols:
        v = a[col].to_numpy()
        ax.bar(x, v, bottom=bottom, width=1.0, color=ARCH_COLORS[col],
               edgecolor="none", label=f"{col}")
        bottom += v
    ax.set_xlim(0, len(a))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(f"Developer (n={len(a)}, sorted by dominant α)", fontsize=8)
    ax.set_ylabel("α (archetype mixture)", fontsize=8)
    ax.set_title("(D) Per-developer archetype mixture α (sorted)", fontsize=10)
    # vertical lines at dominant transitions
    dom = a["dominant"].to_numpy()
    transitions = np.where(dom[:-1] != dom[1:])[0]
    for t in transitions:
        ax.axvline(t + 0.5, color="black", lw=0.4, alpha=0.5)
    # archetype label band
    cur = 0
    for col in arch_cols:
        n = (dom == col).sum()
        if n == 0:
            continue
        ax.text(cur + n / 2, 1.03, col, ha="center", va="bottom",
                fontsize=9, color=ARCH_COLORS[col], fontweight="bold")
        cur += n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile-csv",
                   default="outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_archetype_profile.csv",
                   type=pathlib.Path)
    p.add_argument("--alpha-csv",
                   default="outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_dev_alpha_with_metrics.csv",
                   type=pathlib.Path)
    p.add_argument("--out",
                   default="outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/fig_k4_eugster_profile.pdf",
                   type=pathlib.Path)
    p.add_argument("--signature-only", action="store_true",
                   help="Panel (C) signature features のみを単独 figure として出力")
    p.add_argument("--topn", type=int, default=5,
                   help="signature features の上位 N 件 (default: 5)")
    p.add_argument("--pair", action="store_true",
                   help="ペアarchetype モード（P1..P4 ラベルと PAIR_NAMES を使用）")
    args = p.parse_args()

    df = pd.read_csv(args.profile_csv, index_col=0)
    # 列順を FEATURE_GROUPS に揃える（path 特徴量など未グループ化の列は末尾に維持）
    grouped = [f for _, feats in FEATURE_GROUPS for f in feats if f in df.columns]
    extras = [c for c in df.columns if c not in grouped]
    ordered_feats = grouped + extras
    df = df[ordered_feats]
    print(f"archetypes: {list(df.index)}, features: {len(df.columns)}")

    if args.signature_only:
        fig, ax = plt.subplots(figsize=(14, 12))
        panel_signature(ax, df, topn=args.topn, standalone=True,
                        pair_mode=args.pair)
        fig.savefig(args.out, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {args.out}")
        return

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0],
                          height_ratios=[1.0, 0.95],
                          hspace=0.35, wspace=0.45)

    ax_r = fig.add_subplot(gs[0, 0], projection="polar")
    panel_overlay_radar(ax_r, df, ordered_feats)

    ax_b = fig.add_subplot(gs[:, 1])
    panel_grouped_bars(ax_b, df)

    ax_s = fig.add_subplot(gs[1, 0])
    panel_signature(ax_s, df, topn=5)

    fig.suptitle(
        "k=4 Archetypal analysis of OpenStack reviewers — Eugster-style profile",
        fontsize=12, y=0.995,
    )
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {args.out}")

    # 別ファイルで alpha stack (横長)
    fig2, ax = plt.subplots(figsize=(10, 3.2))
    panel_alpha_stack(ax, args.alpha_csv)
    out2 = args.out.with_name(args.out.stem + "_alphastack.pdf")
    fig2.savefig(out2, format="pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()

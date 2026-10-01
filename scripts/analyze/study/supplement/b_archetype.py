# -*- coding: utf-8 -*-
"""補足資料 B群: 原型分析の深掘り.

入力は既存の原型分析出力 CSV のみ (原型分析そのものは再実行しない)。

実行: .venv/bin/python scripts/analyze/study/supplement/b_archetype.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DIVERGING, FEATURE_GROUPS, GROUP_OF, MODEL_COLORS, MODEL_LABELS, ROOT,
    SOURCES, new_fig, ordered_features, save_fig, save_section, save_table,
    shorten, write_manifest,
)

K_GRID = [2, 3, 4, 5, 6]


def load_profile(src, k: int = 4) -> pd.DataFrame:
    df = pd.read_csv(src.path(src.profile_file, k), index_col=0)
    return df.loc[[f"A{i + 1}" for i in range(k)]]


def load_alpha(src, k: int = 4) -> pd.DataFrame:
    idx = 0 if src.level == "dev" else None
    return pd.read_csv(src.path(src.alpha_file, k), index_col=idx)


def signature(df: pd.DataFrame) -> pd.DataFrame:
    """各原型の「自分の値 − 他原型の単純平均」(archetype_profile_eugster.py:199-209 と同じ定義)。"""
    out = {}
    for a in df.index:
        out[a] = df.loc[a] - df.drop(index=a).mean(axis=0)
    return pd.DataFrame(out)  # 行=特徴量, 列=原型


def check_n_consistency() -> pd.DataFrame:
    """hard_metrics の n と alpha CSV の dominant 件数が一致することを確認する。"""
    rows = []
    for (ds, lv), src in SOURCES.items():
        hard = pd.read_csv(src.path(src.hard_file))
        alpha = load_alpha(src)
        counts = alpha["dominant"].value_counts()
        for _, r in hard.iterrows():
            a = r["archetype"]
            n_hard, n_alpha = int(r[src.n_col]), int(counts.get(a, 0))
            assert n_hard == n_alpha, f"{ds}/{lv} {a}: hard={n_hard} alpha={n_alpha}"
            rows.append({"データセット": src.title, "原型": src.label(a),
                         "n": n_hard})
    print(f"[check] hard_metrics の n と dominant 件数が全{len(rows)}行で一致")
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig_b1_concept() -> None:
    """S-B1: 原型分析の定式化と k-means との違い (概念図)。"""
    rng = np.random.default_rng(7)
    n = 260
    # 三角形の内部に点を散らす (概念図であって実データではない)
    verts = np.array([[0.08, 0.12], [0.95, 0.30], [0.45, 0.95]])
    w = rng.dirichlet([1.6, 1.6, 1.6], size=n)
    pts = w @ verts + rng.normal(0, 0.012, size=(n, 2))

    fig = new_fig(12.8, 6.0)
    ax = fig.add_axes([0.04, 0.10, 0.42, 0.74])
    ax2 = fig.add_axes([0.52, 0.10, 0.44, 0.74])

    ax.scatter(pts[:, 0], pts[:, 1], s=16, color="#BBBBBB", zorder=2)
    # k-means 的な重心 (3分割の平均) と 原型 (端)
    km = np.array([pts[w.argmax(1) == i].mean(axis=0) for i in range(3)])
    ax.scatter(km[:, 0], km[:, 1], s=260, marker="X", color="#4DBBD5",
               edgecolor="white", linewidth=2, zorder=4, label="k-means の重心")
    ax.scatter(verts[:, 0], verts[:, 1], s=300, marker="^", color="#E64B35",
               edgecolor="white", linewidth=2, zorder=5, label="原型（端の極端な型）")
    hull = np.vstack([verts, verts[:1]])
    ax.plot(hull[:, 0], hull[:, 1], color="#E64B35", lw=2, ls="--", zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("概念図：重心は中央に寄る／原型は端に立つ", fontsize=17, pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=13)

    ax2.axis("off")
    ax2.set_title("定式化", fontsize=17, pad=10, loc="left")
    lines = [
        (r"$\min_{\alpha,\beta}\ \|X - \alpha Z\|_F^2,\qquad Z = \beta X$", 0.86, 17),
        (r"$\alpha_{ij}\geq 0,\ \sum_j \alpha_{ij}=1,\qquad "
         r"\beta_{jl}\geq 0,\ \sum_l \beta_{jl}=1$", 0.74, 15),
        ("・$Z$（原型）＝データ点の凸結合 → 必ずデータの外殻（端）に位置する", 0.58, 14),
        ("・$\\alpha$（所属度）＝各ユニットを原型の凸結合で表す重み（合計1）", 0.48, 14),
        ("・ハード割当て＝$\\alpha$ が最大の原型（dominant）", 0.38, 14),
        ("・$K$ は残差平方和 RSS/N のエルボーで決定（本研究は $K=4$）", 0.28, 14),
        ("・実装：交互最適化 max_iter=120, seed=777", 0.16, 13),
    ]
    for text, y, size in lines:
        ax2.text(0.0, y, text, fontsize=size, va="center", color="#333333")

    save_fig(fig, "S-B1", "原型分析の定式化と k-means との違い",
             "原型はデータの凸包上に立つため、平均に埋もれない極端な型を取り出せる。",
             "scripts/analyze/study/archetype_irl_rf.py:75（左は概念図・実データではない）")


def fig_b2_elbow() -> None:
    """S-B2: 4系統のエルボー。"""
    fig = new_fig(12.0, 6.2)
    ax = fig.add_subplot(111)
    styles = {("openstack", "dev"): ("#E64B35", "o", "-"),
              ("openstack", "pair"): ("#E64B35", "s", "--"),
              ("aosp", "dev"): ("#4DBBD5", "o", "-"),
              ("aosp", "pair"): ("#4DBBD5", "s", "--")}
    tab = []
    for key, src in SOURCES.items():
        df = pd.read_csv(src.path(src.elbow_file))
        col = [c for c in df.columns if c.startswith("rss_per")]
        assert len(col) == 1, f"rss_per* 列が特定できない: {df.columns.tolist()}"
        color, marker, ls = styles[key]
        ax.plot(df["k"], df[col[0]], marker=marker, ls=ls, color=color, lw=2.2,
                ms=9, label=src.title)
        for _, r in df.iterrows():
            tab.append({"データセット": src.title, "K": int(r["k"]),
                        "RSS/N": round(float(r[col[0]]), 4)})
    ax.axvline(4, color="#888888", ls=":", lw=2, zorder=1)
    ax.text(4.05, ax.get_ylim()[1] * 0.97, "K = 4 を採用", fontsize=14,
            color="#555555", va="top")
    ax.set_xticks(K_GRID)
    ax.set_xlabel("原型数 K")
    ax.set_ylabel("ユニットあたり残差平方和 RSS/N")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.set_title("原型数 K に対する平均残差平方和（4系統とも K=4 以降で改善が鈍る）", pad=12)
    save_fig(fig, "S-B2", "エルボー法による原型数 K の決定（4系統）",
             "K=4 以降は RSS/N の改善幅が小さくなる。dev/pair・両プロジェクトで同じ傾向。",
             "outputs/**/archetype/**/rss_elbow.csv")
    save_table(pd.DataFrame(tab), "S-B2t", "エルボーの実数値（RSS/N）",
               "図 S-B2 の元数値。", "outputs/**/archetype/**/rss_elbow.csv",
               col_widths=[3.6, 1.4, 2.2], font_size=11)


def fig_signature_heatmap(src, fig_id: str) -> None:
    prof = load_profile(src)
    sig = signature(prof)
    feats = ordered_features(sig.index)
    sig = sig.loc[feats]
    vmax = float(np.abs(sig.to_numpy()).max())

    h = 1.9 + 0.20 * len(feats)
    fig = new_fig(11.0, h)
    ax = fig.add_axes([0.30, 0.06, 0.52, 0.84])
    im = ax.imshow(sig.to_numpy(), cmap=DIVERGING, vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(sig.columns)))
    ax.set_xticklabels([src.label(a).replace(" ", "\n") for a in sig.columns],
                       fontsize=12)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([shorten(f) for f in feats], fontsize=11)
    for i, f in enumerate(feats):
        for j in range(len(sig.columns)):
            v = sig.iloc[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > vmax * 0.55 else "#333333")
    # グループ区切り
    ypos = -0.5
    for gname, gfeats in FEATURE_GROUPS:
        m = len([f for f in gfeats if f in feats])
        if m == 0:
            continue
        ax.axhline(ypos, color="white", lw=3)
        # グループ名は目盛りラベルより外側 (軸座標 x=-0.33) に置く
        ax.text(-0.33, ypos + m / 2, gname, rotation=90, va="center",
                ha="center", fontsize=12, color="#555555",
                transform=ax.get_yaxis_transform())
        ax.plot([-0.30, -0.30], [ypos + 0.1, ypos + m - 0.6], color="#CCCCCC",
                lw=1.6, clip_on=False, transform=ax.get_yaxis_transform())
        ypos += m
    cax = fig.add_axes([0.85, 0.30, 0.02, 0.36])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("その原型の値 − 他原型の平均", fontsize=11)
    ax.set_title(f"{src.title}：全特徴量の「原型 − 他原型平均」", fontsize=17, pad=34)
    save_fig(fig, fig_id, f"全特徴量シグネチャ（{src.title}）",
             "発表スライドは差の大きい上位5特徴量のみ。ここでは全特徴量を載せている。",
             src.rel(src.profile_file))


def fig_signature_bars(src, fig_id: str) -> None:
    prof = load_profile(src)
    sig = signature(prof)
    feats = ordered_features(sig.index)[::-1]   # 上から在籍・量の順になるよう反転
    sig = sig.loc[feats]
    vmax = float(np.abs(sig.to_numpy()).max()) * 1.12

    h = 1.6 + 0.185 * len(feats)
    fig = new_fig(13.2, h)
    axes = fig.subplots(1, len(sig.columns), sharey=True)
    for ax, a in zip(axes, sig.columns):
        ax.barh(range(len(feats)), sig[a].to_numpy(), color=src.color(a),
                height=0.72, zorder=3)
        ax.axvline(0, color="#666666", lw=1)
        ax.set_xlim(-vmax, vmax)
        ax.set_title(src.label(a).replace(" ", "\n"), fontsize=13, pad=8)
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=10)
    axes[0].set_yticks(range(len(feats)))
    axes[0].set_yticklabels([shorten(f) for f in feats], fontsize=10)
    fig.suptitle(f"{src.title}：原型ごとの全特徴量プロファイル（他原型平均との差）",
                 fontsize=17, y=0.995)
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    save_fig(fig, fig_id, f"原型別の全特徴量プロファイル（{src.title}）",
             "横軸は他原型平均との差。棒の長い特徴量がその原型を特徴づける。",
             src.rel(src.profile_file))


def fig_b11_alpha() -> None:
    """S-B11: α の分布 (混合の強さ)。"""
    fig = new_fig(12.6, 6.8)
    axes = fig.subplots(2, 2)
    rows = []
    for ax, (key, src) in zip(axes.ravel(), SOURCES.items()):
        alpha = load_alpha(src)
        cols = [c for c in alpha.columns if c in {"A1", "A2", "A3", "A4"}]
        amax = alpha[cols].max(axis=1)
        ax.hist(amax, bins=np.arange(0.25, 1.0001, 0.05), color="#7E6148",
                edgecolor="white", zorder=3)
        ax.axvline(float(amax.median()), color="#E64B35", lw=2.2, zorder=4)
        ax.text(float(amax.median()) + 0.01, ax.get_ylim()[1] * 0.92,
                f"中央値 {amax.median():.2f}", color="#E64B35", fontsize=12,
                va="top")
        ax.set_title(f"{src.title}（n={len(alpha):,}）", fontsize=14)
        ax.set_xlabel("最大所属度 $\\max_j \\alpha_j$")
        ax.set_ylabel("ユニット数")
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        rows.append({
            "データセット": src.title, "n": len(alpha),
            "α_max 中央値": round(float(amax.median()), 3),
            "α_max ≥ 0.8 の割合": f"{(amax >= 0.8).mean() * 100:.1f}%",
            "α_max < 0.5 の割合": f"{(amax < 0.5).mean() * 100:.1f}%",
        })
    fig.suptitle("所属度αの分布：原型にどれだけ「純粋に」寄っているか", fontsize=17)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    save_fig(fig, "S-B11", "所属度 α の分布",
             "ハード割当て（dominant）は α 最大の原型。α_max が低いユニットは複数原型の混合。",
             "outputs/**/k4_*_alpha_with_metrics.csv")
    save_table(pd.DataFrame(rows), "S-B11t", "所属度 α の要約",
               "図 S-B11 の元数値。", "outputs/**/k4_*_alpha_with_metrics.csv",
               col_widths=[3.2, 1.4, 2.2, 2.6, 2.6], font_size=11)


def fig_b12_k_composition() -> None:
    """S-B12: K=2..6 での原型サイズ構成比。"""
    fig = new_fig(12.6, 6.8)
    axes = fig.subplots(2, 2)
    ramp = ["#D6E8F2", "#9FCBE0", "#63A9CC", "#2E7FA8", "#17607A", "#0B3D4F"]
    for ax, (key, src) in zip(axes.ravel(), SOURCES.items()):
        for k in K_GRID:
            hard = pd.read_csv(src.path(src.hard_file, k))
            n = hard[src.n_col].to_numpy(dtype=float)
            share = n / n.sum()
            left = 0.0
            for i, s in enumerate(share):
                ax.barh(k, s, left=left, height=0.68, color=ramp[i],
                        edgecolor="white", linewidth=1.2, zorder=3)
                if s > 0.07:
                    ax.text(left + s / 2, k, f"{s * 100:.0f}%", ha="center",
                            va="center", fontsize=10,
                            color="white" if i >= 3 else "#333333")
                left += s
        ax.set_yticks(K_GRID)
        ax.set_ylabel("原型数 K")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.set_title(src.title, fontsize=14)
    fig.suptitle("原型数 K を変えたときのサイズ構成比（極端に小さい原型は出ていない）",
                 fontsize=17)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    save_fig(fig, "S-B12", "K を変えたときの原型サイズ構成比",
             "K=4 でも構成比が偏りすぎず、K を増やしても細かく割れていくだけ。",
             "outputs/**/k{2..6}_hard*metrics.csv")


def fig_b13_k_delta() -> None:
    """S-B13: K=2..6 で原型別 Δ(IRL−RF) がどれだけ広がるか。"""
    fig = new_fig(12.6, 6.8)
    axes = fig.subplots(2, 2, sharex=True)
    rows = []
    for ax, (key, src) in zip(axes.ravel(), SOURCES.items()):
        for k in K_GRID:
            pooled = pd.read_csv(src.path(src.pooled_file, k))
            piv = pooled.pivot(index="archetype", columns="model", values="auc")
            delta = (piv["cold_irl"] - piv["rf_dir"]) * 100
            ax.scatter([k] * len(delta), delta.to_numpy(), s=80, zorder=3,
                       color="#E64B35" if k == 4 else "#AAAAAA",
                       edgecolor="white", linewidth=1.2)
            ax.plot([k, k], [delta.min(), delta.max()], lw=2, zorder=2,
                    color="#E64B35" if k == 4 else "#CCCCCC")
            for a, v in delta.items():
                rows.append({"データセット": src.title, "K": k, "原型": a,
                             "Δ(IRL−RF) pp": round(float(v), 2)})
        ax.axhline(0, color="#666666", lw=1.2)
        ax.set_xticks(K_GRID)
        ax.set_title(src.title, fontsize=14)
        ax.set_ylabel("Δ AUC (IRL − RF) [pp]")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("原型数 K")
    fig.suptitle("原型ごとの IRL−RF 差は K によらず現れる（赤＝採用した K=4）",
                 fontsize=17)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    save_fig(fig, "S-B13", "K を変えても原型間の得意不得意は消えない",
             "どの K でも IRL 優位な原型と RF 優位な原型に分かれる。K=4 は恣意的な選択ではない。",
             "outputs/**/k{2..6}_pooled_auc*.csv")
    save_table(pd.DataFrame(rows), "S-B13t", "K別・原型別の Δ(IRL−RF)",
               "図 S-B13 の元数値（pooled AUC の差）。",
               "outputs/**/k{2..6}_pooled_auc*.csv",
               col_widths=[3.2, 1.0, 1.4, 2.6], font_size=10)


def fig_b14_correspondence() -> None:
    """S-B14: OpenStack と AOSP の原型対応 (cosine 類似度)。"""
    fig = new_fig(12.4, 5.8)
    axes = fig.subplots(1, 2)
    rows = []
    for ax, level in zip(axes, ["pair", "dev"]):
        os_src, ao_src = SOURCES[("openstack", level)], SOURCES[("aosp", level)]
        p_os, p_ao = load_profile(os_src), load_profile(ao_src)
        assert list(p_os.columns) == list(p_ao.columns), "特徴量の並びが一致しない"
        A = p_os.to_numpy(); B = p_ao.to_numpy()
        sim = (A @ B.T) / (np.linalg.norm(A, axis=1)[:, None]
                           * np.linalg.norm(B, axis=1)[None, :])
        im = ax.imshow(sim, cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{ao_src.prefix}{i + 1}" for i in range(4)],
                           fontsize=13)
        # AOSP 側の原型名は軸ラベルに入れると重なるので下に凡例として置く
        key = [ao_src.label(f"A{i + 1}") for i in range(4)]
        ax.text(0.5, -0.20, " ／ ".join(key[:2]) + "\n" + " ／ ".join(key[2:]),
                transform=ax.transAxes, ha="center", va="top", fontsize=10,
                color="#555555", linespacing=1.5)
        ax.set_yticks(range(4))
        ax.set_yticklabels([os_src.label(f"A{i + 1}").replace(" ", "\n")
                            for i in range(4)], fontsize=10)
        ax.set_xlabel("AOSP", fontsize=13)
        ax.set_ylabel("OpenStack", fontsize=13)
        for i in range(4):
            j = int(sim[i].argmax())
            for jj in range(4):
                ax.text(jj, i, f"{sim[i, jj]:.3f}", ha="center", va="center",
                        fontsize=11,
                        color="white" if sim[i, jj] > 0.8 else "#333333",
                        fontweight="bold" if jj == j else "normal")
                rows.append({"粒度": level, "OpenStack": os_src.label(f"A{i+1}"),
                             "AOSP": ao_src.label(f"A{jj+1}"),
                             "cosine": round(float(sim[i, jj]), 3),
                             "最近傍": "○" if jj == j else ""})
            ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor="#E64B35", lw=2.5))
        ax.set_title("ペア単位（32次元）" if level == "pair" else "開発者単位（26次元）",
                     fontsize=15, pad=10)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("OpenStack と AOSP の原型プロファイルの対応（cosine 類似度・赤枠＝行の最近傍）",
                 fontsize=16)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    save_fig(fig, "S-B14", "2プロジェクトの原型はどこまで同じか",
             "ペア単位は P1↔P1 / P2↔P2 が対応し、P3・P4は番号がねじれて対応する（配色を入れ替えた根拠）。"
             "開発者単位は1対1に揃わない。",
             "outputs/**/k4_archetype_profile.csv")
    save_table(pd.DataFrame(rows), "S-B14t", "原型対応の cosine 類似度",
               "図 S-B14 の元数値。", "outputs/**/k4_archetype_profile.csv",
               col_widths=[1.2, 3.6, 3.8, 1.6, 1.2], font_size=9)


def fig_b15_dev_vs_pair() -> None:
    """S-B15: 開発者単位の原型 × ペア単位の原型 のクロス集計。"""
    fig = new_fig(12.6, 5.8)
    axes = fig.subplots(1, 2)
    rows = []
    for ax, ds in zip(axes, ["openstack", "aosp"]):
        dev_src, pair_src = SOURCES[(ds, "dev")], SOURCES[(ds, "pair")]
        dev = load_alpha(dev_src)[["dominant"]].rename(
            columns={"dominant": "dev_arch"})
        pair = load_alpha(pair_src)[["developer", "dominant"]].rename(
            columns={"dominant": "pair_arch"})
        merged = pair.merge(dev, left_on="developer", right_index=True, how="inner")
        ct = pd.crosstab(merged["dev_arch"], merged["pair_arch"])
        ct = ct.reindex(index=[f"A{i+1}" for i in range(4)],
                        columns=[f"A{i+1}" for i in range(4)], fill_value=0)
        share = ct.div(ct.sum(axis=1), axis=0)
        im = ax.imshow(share.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"P{i + 1}" for i in range(4)], fontsize=13)
        key = [pair_src.label(c) for c in share.columns]
        ax.text(0.5, -0.20, " ／ ".join(key[:2]) + "\n" + " ／ ".join(key[2:]),
                transform=ax.transAxes, ha="center", va="top", fontsize=10,
                color="#555555", linespacing=1.5)
        ax.set_yticks(range(4))
        ax.set_yticklabels([dev_src.label(i).replace(" ", "\n")
                            for i in share.index], fontsize=10)
        ax.set_xlabel("ペア単位の原型", fontsize=13)
        ax.set_ylabel("開発者単位の原型", fontsize=13)
        for i in range(4):
            for j in range(4):
                v = share.iloc[i, j]
                ax.text(j, i, f"{v * 100:.0f}%\n({ct.iloc[i, j]:,})",
                        ha="center", va="center", fontsize=10,
                        color="white" if v > 0.55 else "#333333")
                rows.append({
                    "プロジェクト": "OpenStack" if ds == "openstack" else "AOSP",
                    "開発者単位": dev_src.label(share.index[i]),
                    "ペア単位": pair_src.label(share.columns[j]),
                    "ペア数": int(ct.iloc[i, j]),
                    "行内比率": f"{v * 100:.1f}%"})
        ax.set_title(f"{'OpenStack' if ds == 'openstack' else 'AOSP'}"
                     f"（対象ペア {len(merged):,}）", fontsize=15, pad=10)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("開発者単位の原型は、ペア単位ではどの原型に分かれるか（行=100%）",
                 fontsize=16)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    save_fig(fig, "S-B15", "開発者単位とペア単位の原型の関係",
             "同じ開発者でもディレクトリごとに違う原型に分かれる＝ペア単位で見る意味がある。",
             "outputs/**/k4_dev_alpha_with_metrics.csv × k4_pair_alpha_with_metrics.csv")
    save_table(pd.DataFrame(rows), "S-B15t", "開発者単位×ペア単位のクロス集計",
               "図 S-B15 の元数値。",
               "outputs/**/k4_*_alpha_with_metrics.csv",
               col_widths=[1.8, 3.4, 3.4, 1.4, 1.4], font_size=9)


def main() -> None:
    n_tab = check_n_consistency()
    save_section("S-B0", "B. 原型分析の深掘り",
                 "原型の定義・K の決め方・全特徴量プロファイル・2プロジェクトの対応")
    fig_b1_concept()
    fig_b2_elbow()
    save_table(n_tab, "S-B2n", "原型ごとのユニット数（K=4）",
               "ハード割当て（α が最大の原型）での件数。合計は OpenStack 289人/3,677ペア、"
               "AOSP 1,248人/21,291ペア。",
               "outputs/**/k4_hard*metrics.csv（alpha CSV の dominant 件数と一致を確認）",
               col_widths=[3.4, 5.0, 1.6], font_size=11)
    for (ds, lv), fid in [(("openstack", "pair"), "S-B3"),
                          (("aosp", "pair"), "S-B4"),
                          (("openstack", "dev"), "S-B5"),
                          (("aosp", "dev"), "S-B6")]:
        fig_signature_heatmap(SOURCES[(ds, lv)], fid)
    for (ds, lv), fid in [(("openstack", "pair"), "S-B7"),
                          (("aosp", "pair"), "S-B8"),
                          (("openstack", "dev"), "S-B9"),
                          (("aosp", "dev"), "S-B10")]:
        fig_signature_bars(SOURCES[(ds, lv)], fid)
    fig_b11_alpha()
    fig_b12_k_composition()
    fig_b13_k_delta()
    fig_b14_correspondence()
    fig_b15_dev_vs_pair()
    write_manifest("b")


if __name__ == "__main__":
    main()

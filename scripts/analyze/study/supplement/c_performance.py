# -*- coding: utf-8 -*-
"""補足資料 C群: 予測性能の深掘り (原型別・期間別・アンサンブル・アブレーション).

先に c_period_auc.py を実行して tab/period_auc_*.csv を作っておくこと。

実行: .venv/bin/python scripts/analyze/study/supplement/c_performance.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DIVERGING, HORIZON_LABELS, HORIZONS, MODEL_COLORS, MODEL_LABELS, ROOT,
    SOURCES, TAB, new_fig, save_fig, save_section, save_table, write_manifest,
)

ARCH_IDS = ["A1", "A2", "A3", "A4"]
XLSX = ROOT / "docs" / "SES発表用_アブレーション.xlsx"
HEADLINE = ROOT / "outputs" / "ses_review" / "b3_ablation"
PERIOD_SRC = {
    ("openstack", "pair"): "outputs/analysis/ses_supplement/tab/period_auc_openstack_pair.csv"
                           "（pooled_pairs_blabel_t0_20230101.csv から今回算出）",
    ("openstack", "dev"): "outputs/analysis/ses_supplement/tab/period_auc_openstack_dev.csv"
                          "（pooled_pairs_blabel_t0_20230101.csv から今回算出）",
    ("aosp", "pair"): "outputs/aosp32/archetype/pair/k4_period_auc.csv",
    ("aosp", "dev"): "outputs/aosp32/archetype/dev/k4_period_auc.csv",
}


def period_df(ds: str, level: str) -> pd.DataFrame:
    return pd.read_csv(TAB / f"period_auc_{ds}_{level}.csv")


def _arch_yticks(ax, src) -> None:
    """原型名は長いので軸には ID だけ置き、名前は図の下に凡例として出す。"""
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{src.prefix}{i + 1}" for i in range(4)], fontsize=14)
    key = [src.label(a) for a in ARCH_IDS]
    ax.text(0.5, -0.13, " ／ ".join(key[:2]) + "\n" + " ／ ".join(key[2:]),
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            color="#555555", linespacing=1.5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig_period_heatmap(level: str, fig_id: str) -> None:
    """原型 × 評価期間 の Δ(IRL − RF) ヒートマップ。"""
    data = {ds: period_df(ds, level) for ds in ("openstack", "aosp")}
    vmax = max(float(np.abs(d["delta_pp"]).max()) for d in data.values())

    fig = new_fig(13.0, 6.2)
    axes = fig.subplots(1, 2)
    rows = []
    for ax, ds in zip(axes, ["openstack", "aosp"]):
        src = SOURCES[(ds, level)]
        d = data[ds]
        mat = d.pivot(index="archetype", columns="horizon",
                      values="delta_pp").reindex(index=ARCH_IDS, columns=HORIZONS)
        nmap = d.pivot(index="archetype", columns="horizon",
                       values="n").reindex(index=ARCH_IDS, columns=HORIZONS)
        im = ax.imshow(mat.to_numpy(), cmap=DIVERGING, vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels([HORIZON_LABELS[h] for h in HORIZONS], fontsize=13)
        _arch_yticks(ax, src)
        for i, a in enumerate(ARCH_IDS):
            for j, h in enumerate(HORIZONS):
                v = mat.loc[a, h]
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=14,
                        fontweight="bold",
                        color="white" if abs(v) > vmax * 0.55 else "#333333")
                rows.append({
                    "データセット": src.title, "原型": src.label(a),
                    "期間": HORIZON_LABELS[h], "n": int(nmap.loc[a, h]),
                    "IRL": round(float(d[(d.archetype == a) & (d.horizon == h)]
                                       ["auc_cold_irl"].iloc[0]), 4),
                    "RF": round(float(d[(d.archetype == a) & (d.horizon == h)]
                                      ["auc_rf_dir"].iloc[0]), 4),
                    "Δ(pp)": round(float(v), 2)})
        n_unit = int(nmap.iloc[:, 0].sum())
        ax.set_title(f"{'OpenStack' if ds == 'openstack' else 'AOSP'}"
                     f"（各期間 計 {n_unit:,} 行）", fontsize=15, pad=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Δ AUC ＝ IRL − RF [pp]（赤＝IRL優位）", fontsize=12)
    lvl = "ペア単位" if level == "pair" else "開発者単位"
    fig.suptitle(f"原型 × 評価期間ごとの IRL と RF の優劣（{lvl}）", fontsize=17)
    save_fig(fig, fig_id, f"原型×期間の Δ(IRL−RF)（{lvl}）",
             "発表スライドでは P1 と P3/P4 の2行だけを見せている。ここでは全原型×全期間。",
             f"{PERIOD_SRC[('openstack', level)]} ／ {PERIOD_SRC[('aosp', level)]}")
    save_table(pd.DataFrame(rows), f"{fig_id}t", f"原型×期間の AUC 実数値（{lvl}）",
               "図の元数値。IRL・RF は pooled AUC（B定義ラベル）。",
               f"{PERIOD_SRC[('openstack', level)]} ／ {PERIOD_SRC[('aosp', level)]}",
               col_widths=[2.6, 3.4, 1.2, 1.2, 1.2, 1.2, 1.4], font_size=9)


def fig_c3_ens_gain() -> None:
    """S-C3: アンサンブルがどの原型・どの期間で効くか (OpenStack)。"""
    fig = new_fig(12.4, 6.0)
    axes = fig.subplots(1, 2)
    vmax = 0.0
    mats = {}
    for level in ["pair", "dev"]:
        d = period_df("openstack", level)
        mats[level] = d.pivot(index="archetype", columns="horizon",
                              values="ens_gain_pp").reindex(index=ARCH_IDS,
                                                            columns=HORIZONS)
        vmax = max(vmax, float(np.abs(mats[level].to_numpy()).max()))
    for ax, level in zip(axes, ["pair", "dev"]):
        src = SOURCES[("openstack", level)]
        im = ax.imshow(mats[level].to_numpy(), cmap=DIVERGING, vmin=-vmax,
                       vmax=vmax, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels([HORIZON_LABELS[h] for h in HORIZONS], fontsize=13)
        _arch_yticks(ax, src)
        for i, a in enumerate(ARCH_IDS):
            for j, h in enumerate(HORIZONS):
                v = mats[level].loc[a, h]
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=14,
                        fontweight="bold",
                        color="white" if abs(v) > vmax * 0.55 else "#333333")
        ax.set_title("ペア単位" if level == "pair" else "開発者単位", fontsize=15,
                     pad=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("アンサンブル − 単体ベスト [pp]", fontsize=12)
    fig.suptitle("アンサンブルの利得は原型・期間で偏る（OpenStack）", fontsize=17)
    save_fig(fig, "S-C3", "アンサンブルはどこで効いたか（OpenStack）",
             "IRLとRFの優劣が入れ替わる原型ほど平均の効果が大きい。負の原型では単体の方が良い。",
             "outputs/analysis/ses_supplement/tab/period_auc_openstack_{pair,dev}.csv"
             "（AOSP はラベル定義が既存CSVと異なるため未算出）")


def fig_forest(level: str, fig_id: str) -> None:
    """原型別 pooled AUC と 95%CI。"""
    fig = new_fig(13.0, 6.4)
    axes = fig.subplots(1, 2, sharex=True)
    rows = []
    for ax, ds in zip(axes, ["openstack", "aosp"]):
        src = SOURCES[(ds, level)]
        pooled = pd.read_csv(src.path(src.pooled_file))
        models = [m for m in ["cold_irl", "rf_dir", "ens"]
                  if m in set(pooled["model"])]
        for i, a in enumerate(ARCH_IDS):
            for j, m in enumerate(models):
                r = pooled[(pooled.archetype == a) & (pooled.model == m)].iloc[0]
                y = -(i * (len(models) + 1) + j)
                ax.errorbar(r["auc"], y,
                            xerr=[[r["auc"] - r["auc_ci_lo"]],
                                  [r["auc_ci_hi"] - r["auc"]]],
                            fmt="o", ms=9, lw=2.4, capsize=5,
                            color=MODEL_COLORS[m],
                            label=MODEL_LABELS[m] if i == 0 else None)
                ax.text(r["auc_ci_hi"] + 0.006, y, f"{r['auc']:.3f}", fontsize=11,
                        va="center", color="#333333")
                rows.append({
                    "データセット": src.title, "原型": src.label(a),
                    "モデル": MODEL_LABELS[m], "n": int(r["n_samples"]),
                    "AUC": round(float(r["auc"]), 4),
                    "95%CI": f"[{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}]"})
        ticks = [-(i * (len(models) + 1) + (len(models) - 1) / 2)
                 for i in range(4)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{src.prefix}{i + 1}" for i in range(4)], fontsize=14)
        key = [src.label(a) for a in ARCH_IDS]
        ax.text(0.5, -0.14, " ／ ".join(key[:2]) + "\n" + " ／ ".join(key[2:]),
                transform=ax.transAxes, ha="center", va="top", fontsize=10,
                color="#555555", linespacing=1.5)
        ax.set_xlabel("pooled AUC（95%信頼区間）")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_title(f"{'OpenStack' if ds == 'openstack' else 'AOSP'}", fontsize=15)
    lvl = "ペア単位" if level == "pair" else "開発者単位"
    fig.suptitle(f"原型別の pooled AUC と信頼区間（{lvl}）", fontsize=17)
    # 凡例はデータ点に重ならないよう図の上部に1つだけ置く
    handles = [Line2D([], [], color=MODEL_COLORS[m], marker="o", ms=9, lw=2.4,
                      label=MODEL_LABELS[m] + ("（AOSPのみ）" if m == "ens" else ""))
               for m in ["cold_irl", "rf_dir", "ens"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=3, frameon=False, fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 0.90))
    save_fig(fig, fig_id, f"原型別 pooled AUC と 95%CI（{lvl}）",
             "信頼区間が重なる原型では IRL と RF の差は言い切れない。",
             f"{SOURCES[('openstack', level)].rel(SOURCES[('openstack', level)].pooled_file)}"
             f" ／ {SOURCES[('aosp', level)].rel(SOURCES[('aosp', level)].pooled_file)}"
             "（OpenStack 側の warm_irl は除外）")
    save_table(pd.DataFrame(rows), f"{fig_id}t", f"原型別 pooled AUC の実数値（{lvl}）",
               "図の元数値（bootstrap 95%信頼区間）。", "outputs/**/k4_pooled_auc*.csv",
               col_widths=[2.6, 3.4, 2.0, 1.4, 1.2, 2.4], font_size=10)


def table_wilcoxon(level: str, fig_id: str) -> None:
    """原型別 Wilcoxon 検定（IRL vs RF）。"""
    rows = []
    for ds in ["openstack", "aosp"]:
        src = SOURCES[(ds, level)]
        w = pd.read_csv(src.path(src.wilcoxon_file))
        if "metric" in w.columns:              # OpenStack dev のみ acc/auc 両方持つ
            w = w[w["metric"] == "auc"]
        w = w[w["comparison"].str.contains("cold")]
        n_col = [c for c in w.columns if c.startswith("n_")][0]
        med_col = [c for c in w.columns if c.startswith("median")][0]
        for _, r in w.iterrows():
            p = float(r["p_value"])
            rows.append({
                "データセット": src.title,
                "原型": src.label(r["archetype"]),
                "n（AUC算出可）": int(r[n_col]),
                "中央値差 IRL−RF": round(float(r[med_col]), 4),
                "W統計量": round(float(r["wilcoxon_stat"]), 1),
                "p値": f"{p:.2e}" if p < 0.001 else f"{p:.3f}",
                "有意": "**" if p < 0.01 else ("*" if p < 0.05 else "n.s."),
            })
    lvl = "ペア単位" if level == "pair" else "開発者単位"
    save_table(pd.DataFrame(rows), fig_id,
               f"原型別 Wilcoxon 符号順位検定（IRL vs RF・{lvl}）",
               "ユニットごとの AUC の対応比較。AUC が計算できる"
               "（正例と負例が両方ある）ユニットのみが対象なので n は原型サイズより小さい。",
               f"{SOURCES[('openstack', level)].rel(SOURCES[('openstack', level)].wilcoxon_file)}"
               f" ／ {SOURCES[('aosp', level)].rel(SOURCES[('aosp', level)].wilcoxon_file)}"
               "（OpenStack の warm_irl 比較・acc 比較の行は除外）",
               col_widths=[2.6, 3.4, 1.8, 2.0, 1.6, 1.6, 1.0], font_size=10)


def fig_c8_ablation() -> None:
    """S-C8: アブレーション (RF / LSTM / IRL の統合AUC)。"""
    # OpenStack は headline CSV から再計算値を読む
    irl = pd.read_csv(HEADLINE / "headline_blabel_irl_main.csv").set_index("horizon")
    lstm = pd.read_csv(HEADLINE / "headline_blabel_lstm_final_only.csv").set_index("horizon")
    os_vals = {"RF（時系列なし）": float(irl.loc["pooled", "RF_roc"]),
               "LSTM（時系列のみ）": float(lstm.loc["pooled", "IRL_roc"]),
               "IRL（時系列＋報酬関数）": float(irl.loc["pooled", "IRL_roc"])}
    # AOSP は対応する CSV がローカルに無いため発表用 xlsx から転記
    x = pd.read_excel(XLSX, header=None)
    ao_vals = {"RF（時系列なし）": float(x.iloc[1, 2]),
               "LSTM（時系列のみ）": float(x.iloc[2, 2]),
               "IRL（時系列＋報酬関数）": float(x.iloc[3, 2])}
    # xlsx の OpenStack 列と CSV 由来の値が一致することを確認
    for i, k in enumerate(os_vals):
        assert abs(float(x.iloc[i + 1, 1]) - round(os_vals[k], 3)) < 1e-9, \
            f"xlsx と CSV が不一致: {k}"

    fig = new_fig(12.4, 6.0)
    axes = fig.subplots(1, 2, sharey=True)
    colors = ["#4DBBD5", "#7E6148", "#E64B35"]
    rows = []
    for ax, (name, vals, note) in zip(
            axes,
            [("OpenStack", os_vals, "headline_blabel_*.csv から算出"),
             ("AOSP", ao_vals, "発表用 xlsx / 論文表から転記（対応CSVなし）")]):
        bars = ax.bar(range(3), list(vals.values()), color=colors, width=0.62,
                      zorder=3)
        for b, v in zip(bars, vals.values()):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", fontsize=15, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels([k.replace("（", "\n（") for k in vals], fontsize=12)
        ax.set_ylim(0.60, 0.88)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"{name}（{note}）", fontsize=14)
        for k, v in vals.items():
            rows.append({"データセット": name, "モデル": k, "AUC（4期間統合）": round(v, 4),
                         "出所": note})
    axes[0].set_ylabel("AUC-ROC（4評価期間統合）")
    fig.suptitle("時系列の符号化と報酬関数としての定式化、効いたのはどちらか", fontsize=17)
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    save_fig(fig, "S-C8", "アブレーション：LSTM と IRL の比較",
             "OpenStack は報酬関数化で +3.8pp。AOSP は逆に LSTM 単体が最も高く、補完効果はエコシステム依存。",
             "outputs/ses_review/b3_ablation/headline_blabel_*.csv ／ "
             "docs/SES発表用_アブレーション.xlsx（AOSP）")
    save_table(pd.DataFrame(rows), "S-C8t", "アブレーションの実数値",
               "図 S-C8 の元数値。AOSP は対応する出力CSVがローカルに存在しないため転記。",
               "outputs/ses_review/b3_ablation/headline_blabel_*.csv ／ "
               "docs/SES発表用_アブレーション.xlsx",
               col_widths=[2.4, 3.6, 2.4, 4.6], font_size=11)


def fig_c9_ensemble() -> None:
    """S-C9: 期間別のアンサンブル効果。"""
    fig = new_fig(13.0, 6.2)
    axes = fig.subplots(1, 2, sharey=True)
    rows = []
    for ax, (ds, note) in zip(axes, [("openstack", "B定義ラベル"),
                                     ("aosp", "焼き込みラベル")]):
        d = pd.read_csv(TAB / f"overall_auc_{ds}.csv")
        xs = np.arange(len(d))
        w = 0.26
        for off, col, key in [(-w, "auc_irl", "cold_irl"), (0, "auc_rf", "rf_dir"),
                              (w, "auc_ens", "ens")]:
            ax.bar(xs + off, d[col], width=w, color=MODEL_COLORS[key],
                   label=MODEL_LABELS[key], zorder=3)
        for i, r in d.iterrows():
            ax.text(i + w, r["auc_ens"] + 0.004, f"{r['ens_gain_pp']:+.1f}pp",
                    ha="center", fontsize=11, color="#333333")
        ax.set_xticks(xs)
        ax.set_xticklabels([HORIZON_LABELS.get(h, "全体") for h in d["horizon"]],
                           fontsize=13)
        ax.set_ylim(0.60, 0.90)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"{'OpenStack' if ds == 'openstack' else 'AOSP'}（{note}）",
                     fontsize=15)
        for _, r in d.iterrows():
            rows.append({
                "データセット": "OpenStack" if ds == "openstack" else "AOSP",
                "期間": HORIZON_LABELS.get(r["horizon"], "全体"),
                "n": int(r["n"]), "正例": int(r["pos"]),
                "IRL": round(r["auc_irl"], 4), "RF": round(r["auc_rf"], 4),
                "アンサンブル": round(r["auc_ens"], 4),
                "単体ベストとの差(pp)": round(r["ens_gain_pp"], 2)})
    axes[0].set_ylabel("AUC-ROC")
    axes[0].legend(frameon=False, ncol=3, loc="upper right", fontsize=12)
    fig.suptitle("IRLとRFの確率を単純平均したときの期間別の効果（数値＝単体ベストとの差）",
                 fontsize=16)
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    save_fig(fig, "S-C9", "アンサンブルの期間別内訳",
             "短期ではIRLの上積みが大きく、長期では単体ベストとほぼ同じか下回る期間もある。",
             "outputs/analysis/ses_supplement/tab/overall_auc_{openstack,aosp}.csv"
             "（OpenStack は headline CSV と一致を確認済み）")
    save_table(pd.DataFrame(rows), "S-C9t", "アンサンブルの実数値",
               "図 S-C9 の元数値。AOSP は pair_predictions の焼き込みラベルから再計算"
               "（論文の表と一致）。",
               "outputs/analysis/ses_supplement/tab/overall_auc_*.csv",
               col_widths=[2.2, 1.4, 1.6, 1.4, 1.4, 1.4, 1.8, 2.2], font_size=10)


def fig_c10_irl_formulation() -> None:
    """S-C10: IRL（MCE-IRL）の定式化。式と処理の流れを 1 枚にまとめる。

    図中の式・数値はすべて実装から転記する（出所は save_fig の source 参照）。
    """
    fig = new_fig(12.8, 6.6)
    ax = fig.add_axes([0.02, 0.04, 0.40, 0.86])
    ax2 = fig.add_axes([0.46, 0.03, 0.53, 0.87])
    for a in (ax, ax2):
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.axis("off")

    # ── 左: 処理の流れ ────────────────────────────────
    ax.set_title("処理の流れ（1 ステップ ＝ 1 か月）", fontsize=17, pad=18)

    def box(y, h, text, face, edge, size=14, bold=False):
        ax.add_patch(FancyBboxPatch(
            (0.06, y), 0.88, h, boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=face, edgecolor=edge, linewidth=1.8, zorder=3))
        ax.text(0.5, y + h / 2, text, ha="center", va="center", fontsize=size,
                color="#333333", zorder=4,
                fontweight="bold" if bold else "normal")

    def arrow(y_from, y_to):
        ax.annotate("", xy=(0.5, y_to), xytext=(0.5, y_from),
                    arrowprops=dict(arrowstyle="-|>", color="#888888", lw=2.0))

    # 月次の状態系列 (3 ステップ分を並べる)
    for i, lab in enumerate([r"$s_{t-2},a_{t-2}$", r"$s_{t-1},a_{t-1}$",
                             r"$s_{t},a_{t}$"]):
        x = 0.06 + i * 0.30
        ax.add_patch(FancyBboxPatch(
            (x, 0.79), 0.28, 0.10,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            facecolor="#F2F2F2", edgecolor="#AAAAAA", linewidth=1.5, zorder=3))
        ax.text(x + 0.14, 0.84, lab, ha="center", va="center", fontsize=13,
                color="#333333", zorder=4)
        if i < 2:
            ax.annotate("", xy=(x + 0.30, 0.84), xytext=(x + 0.285, 0.84),
                        arrowprops=dict(arrowstyle="-|>", color="#AAAAAA", lw=1.6))
    ax.text(0.5, 0.945, "状態 32 次元（開発者状態 26 ＋ ディレクトリ親和度 6）\n"
                        "＋ 行動特徴 5 次元",
            ha="center", va="center", fontsize=11, color="#777777",
            linespacing=1.5)

    arrow(0.78, 0.705)
    box(0.60, 0.10, "Bi-LSTM（2 層）", "#EAF4F9", "#4DBBD5")
    arrow(0.59, 0.475)
    box(0.37, 0.10, r"報酬ヘッド MLP（2 層）→ $R_\theta(s,0),\ R_\theta(s,1)$",
        "#FBE9E5", "#E64B35")
    arrow(0.36, 0.255)
    box(0.15, 0.10, "softmax（Boltzmann 方策）", "#F2EEEA", "#7E6148")
    arrow(0.14, 0.075)
    ax.text(0.5, 0.035, r"継続確率 $\pi_\theta(a{=}1\,|\,s_t)$",
            ha="center", va="center", fontsize=15, color="#333333")

    # ── 右: 定式化 ────────────────────────────────────
    ax2.set_title("定式化（Maximum Causal Entropy IRL）", fontsize=17, pad=8,
                  loc="left")
    lines = [
        ("・状態 $s_t$：その月までの月次集約特徴量", 0.935, 14),
        ("・行動 $a_t \\in \\{0,1\\}$：月 $t$ に当該ディレクトリでレビュー応答したか",
         0.865, 14),
        ("・報酬 $R_\\theta(s,a)$：Bi-LSTM ＋ MLP（特徴量の線形和ではない）",
         0.795, 14),
        ("方策（Boltzmann）", 0.70, 13),
        (r"$\pi_\theta(a\,|\,s_t)=\dfrac{\exp R_\theta(s_t,a)}"
         r"{\sum_{a'}\exp R_\theta(s_t,a')}$", 0.595, 19),
        ("学習目的：expert 軌跡の重み付き対数尤度を最大化", 0.495, 13),
        (r"$\mathcal{L}(\theta)=-\dfrac{\sum_{\tau}\sum_{t} w_t\,m_t\,"
         r"\log \pi_\theta(a_t\,|\,s_t)}{\sum_{\tau}\sum_{t} m_t}$", 0.345, 19),
        ("　$m_t$：系列長パディングのマスク（softmax 交差エントロピーと同値）",
         0.255, 12),
        ("・サンプル重み $w_t$：レビュー依頼あり 1.0 ／ 依頼なし（弱い負例）0.5",
         0.175, 13),
        ("・推論：最終ステップの $\\pi_\\theta(a{=}1|s_T)$ を継続確率として使う",
         0.105, 13),
        ("・実装：epochs=150 / 学習率 3e-4 / 乱数シード 777", 0.035, 13),
    ]
    for text, y, size in lines:
        ax2.text(0.0, y, text, fontsize=size, va="center", color="#333333")

    save_fig(fig, "S-C10", "IRL（MCE-IRL）の定式化",
             "報酬は特徴量の線形和ではなく系列を通した非線形関数。"
             "学習は行動履歴（expert 軌跡）の重み付き対数尤度最大化。",
             "src/review_predictor/IRL/model/mce_irl_predictor.py:10-26,:83-87,"
             ":1039-1068 / scripts/train/train_mce_irl.py:630-637")


def main() -> None:
    save_section("S-C0", "C. 予測性能の深掘り",
                 "原型別・期間別の優劣、信頼区間、検定、アブレーション、アンサンブル")
    fig_c10_irl_formulation()
    fig_period_heatmap("pair", "S-C1")
    fig_period_heatmap("dev", "S-C2")
    fig_c3_ens_gain()
    fig_forest("pair", "S-C4")
    fig_forest("dev", "S-C5")
    table_wilcoxon("pair", "S-C6")
    table_wilcoxon("dev", "S-C7")
    fig_c8_ablation()
    fig_c9_ensemble()
    write_manifest("c")


if __name__ == "__main__":
    main()

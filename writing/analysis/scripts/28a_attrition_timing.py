"""28a: 脱落 (attrition) timing の具体化 — 柱B。

定義:
  attrition_rate(transition) = P(前区間 h_prev で 1 つでも承諾していたペアが
    次区間 h_curr で 1 つも承諾しなくなる割合)
  = n_attrited / n_active_prev

3 遷移:
  T1 = 0-3m  -> 3-6m
  T2 = 3-6m  -> 6-9m
  T3 = 6-9m  -> 9-12m

やること:
  1. 全体: 3 遷移別の重み付き attrition rate (n_active_prev 重み)
  2. dir 別 peak timing 分布 (序盤 T1 / 中盤 T2 / 終盤 T3)
  3. 統計的に有意な (sig_mc) dir の 3 遷移 attrition rate を表で
  4. 図: dir x 3 遷移 の attrition rate ヒートマップ
"""

import json
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("/Users/kazuki-h/Master_research/writing/analysis/results")
HORIZON = RESULTS / "horizon_dir_attrition.csv"
TIMING = RESULTS / "interval_attrition_timing_test.csv"
MASTER = RESULTS / "directory_master.csv"

TRANS_ORDER = ["0-3m→3-6m", "3-6m→6-9m", "6-9m→9-12m"]
TRANS_LABEL = {
    "0-3m→3-6m": "序盤\n0-3m→3-6m",
    "3-6m→6-9m": "中盤\n3-6m→6-9m",
    "6-9m→9-12m": "終盤\n6-9m→9-12m",
}
PHASE_OF = {"0-3m→3-6m": "序盤", "3-6m→6-9m": "中盤", "6-9m→9-12m": "終盤"}


def load():
    h = pd.read_csv(HORIZON)
    h["transition"] = h["h_prev"].str.strip() + "→" + h["h_curr"].str.strip()
    # active が無い遷移 (n_active_prev==0) は除外
    h = h[h["n_active_prev"] > 0].copy()
    t = pd.read_csv(TIMING)
    return h, t


def overall_weighted(h):
    """3 遷移別の重み付き attrition rate (全 dir 集約)。"""
    rows = []
    for tr in TRANS_ORDER:
        sub = h[h["transition"] == tr]
        n_active = sub["n_active_prev"].sum()
        n_attr = sub["n_attrited"].sum()
        rate = n_attr / n_active if n_active > 0 else np.nan
        rows.append(
            {
                "transition": tr,
                "phase": PHASE_OF[tr],
                "n_active_prev_total": int(n_active),
                "n_attrited_total": int(n_attr),
                "weighted_attrition_rate": round(rate, 4),
                "n_dirs": int(sub["directory"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def dir_transition_matrix(h):
    """dir x transition の attrition_rate と n_active を wide で返す。"""
    rate = h.pivot_table(
        index="directory", columns="transition", values="attrition_rate", aggfunc="first"
    )
    nact = h.pivot_table(
        index="directory", columns="transition", values="n_active_prev", aggfunc="first"
    )
    for tr in TRANS_ORDER:
        if tr not in rate.columns:
            rate[tr] = np.nan
            nact[tr] = np.nan
    return rate[TRANS_ORDER], nact[TRANS_ORDER]


def main():
    h, t = load()

    # ---- 1 & 2. 全体重み付き ----
    overall = overall_weighted(h)
    overall_path = RESULTS / "attrition_timing_overall.csv"
    overall.to_csv(overall_path, index=False)
    print("=== 全体 3 遷移別 重み付き attrition rate ===")
    print(overall.to_string(index=False))

    rate, nact = dir_transition_matrix(h)

    # ---- 3. peak timing 分布 (timing test の 34 dir を基準) ----
    # peak_transition は timing test ファイルにある (3 遷移とも観測できた 34 dir)
    t = t.copy()
    t["peak_phase"] = t["peak_transition"].map(PHASE_OF)
    peak_dist = (
        t.groupby("peak_phase").size().reindex(["序盤", "中盤", "終盤"]).fillna(0).astype(int)
    )
    sig_peak_dist = (
        t[t["sig_mc"]]
        .groupby("peak_phase")
        .size()
        .reindex(["序盤", "中盤", "終盤"])
        .fillna(0)
        .astype(int)
    )
    print("\n=== peak timing 分布 (34 dir, 3 遷移とも観測) ===")
    for ph in ["序盤", "中盤", "終盤"]:
        print(f"  {ph}: 全 {peak_dist[ph]} dir / うち有意(sig_mc) {sig_peak_dist[ph]} dir")
    n_sig = int(t["sig_mc"].sum())
    print(f"  有意 dir 合計: {n_sig}/{len(t)}")

    peak_dist_df = pd.DataFrame(
        {
            "peak_phase": ["序盤", "中盤", "終盤"],
            "n_dirs_all": [peak_dist[p] for p in ["序盤", "中盤", "終盤"]],
            "n_dirs_sig_mc": [sig_peak_dist[p] for p in ["序盤", "中盤", "終盤"]],
        }
    )
    peak_dist_path = RESULTS / "attrition_timing_peak_dist.csv"
    peak_dist_df.to_csv(peak_dist_path, index=False)

    # ---- 4. 有意 dir の 3 遷移 attrition rate 表 ----
    sig_dirs = t[t["sig_mc"]].sort_values("attr_range", ascending=False)
    detail_rows = []
    for _, r in sig_dirs.iterrows():
        d = r["directory"]
        rr = rate.loc[d] if d in rate.index else pd.Series({tr: np.nan for tr in TRANS_ORDER})
        nn = nact.loc[d] if d in nact.index else pd.Series({tr: np.nan for tr in TRANS_ORDER})
        detail_rows.append(
            {
                "directory": d,
                "peak_transition": r["peak_transition"],
                "peak_phase": r["peak_phase"],
                "attr_range_pp": round(r["attr_range"] * 100, 1),
                "p_montecarlo": r["p_montecarlo"],
                "rate_T1_序盤": round(rr["0-3m→3-6m"] * 100, 1),
                "rate_T2_中盤": round(rr["3-6m→6-9m"] * 100, 1),
                "rate_T3_終盤": round(rr["6-9m→9-12m"] * 100, 1),
                "n_active_T1": int(nn["0-3m→3-6m"]) if not np.isnan(nn["0-3m→3-6m"]) else 0,
                "n_active_T2": int(nn["3-6m→6-9m"]) if not np.isnan(nn["3-6m→6-9m"]) else 0,
                "n_active_T3": int(nn["6-9m→9-12m"]) if not np.isnan(nn["6-9m→9-12m"]) else 0,
            }
        )
    detail = pd.DataFrame(detail_rows)
    detail_path = RESULTS / "attrition_timing_sig_detail.csv"
    detail.to_csv(detail_path, index=False)
    print("\n=== 有意 (sig_mc) dir の 3 遷移 attrition rate [%] ===")
    print(
        detail[
            [
                "directory",
                "peak_phase",
                "attr_range_pp",
                "rate_T1_序盤",
                "rate_T2_中盤",
                "rate_T3_終盤",
                "p_montecarlo",
            ]
        ].to_string(index=False)
    )

    # ---- 5. ヒートマップ ----
    # 有意 8 dir + range 上位で 12 dir まで補完 (合計 12)
    n_show = 12
    show_dirs = list(sig_dirs["directory"])
    for d in t.sort_values("attr_range", ascending=False)["directory"]:
        if len(show_dirs) >= n_show:
            break
        if d not in show_dirs:
            show_dirs.append(d)
    show_dirs = show_dirs[:n_show]

    mat = rate.loc[[d for d in show_dirs if d in rate.index]].copy()
    mat = mat.reindex(columns=TRANS_ORDER)
    # 行ラベル: dir 名 + (sig)
    sig_set = set(sig_dirs["directory"])
    ylabels = [f"{d}  ★" if d in sig_set else d for d in mat.index]

    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(mat) + 1.8))
    data = mat.values.astype(float) * 100
    im = ax.imshow(data, aspect="auto", cmap="Reds", vmin=0, vmax=100)

    ax.set_xticks(range(len(TRANS_ORDER)))
    ax.set_xticklabels([TRANS_LABEL[tr] for tr in TRANS_ORDER], fontsize=10)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(ylabels, fontsize=9)

    # セルに数値
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
            else:
                color = "white" if v > 55 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("脱落率 [%]", fontsize=10)
    ax.set_title(
        "ディレクトリ別 脱落 timing (★=モンテカルロ検定で有意)\n"
        "同一 dir でも遷移ごとに脱落の山の位置が異なる",
        fontsize=11,
    )
    ax.set_xlabel("遷移 (前区間→次区間)", fontsize=10)
    fig.tight_layout()
    fig_path = RESULTS / "fig_dir_attrition_timing_detail.pdf"
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n図を保存: {fig_path}")

    # ---- 補足: 全体重み付き 棒グラフ ----
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.6))
    vals = overall["weighted_attrition_rate"].values * 100
    bars = ax2.bar(
        range(3),
        vals,
        color=["#9ecae1", "#fdae6b", "#de2d26"],
    )
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([TRANS_LABEL[tr] for tr in TRANS_ORDER], fontsize=10)
    ax2.set_ylabel("重み付き脱落率 [%]", fontsize=10)
    ax2.set_title("全 dir 集約: 遷移が進むほど脱落率が上昇", fontsize=11)
    for b, v, n in zip(bars, vals, overall["n_active_prev_total"]):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + 0.8,
            f"{v:.1f}%\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_ylim(0, max(vals) * 1.25)
    fig2.tight_layout()
    fig2_path = RESULTS / "fig_attrition_timing_overall.pdf"
    fig2.savefig(fig2_path, format="pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"図を保存: {fig2_path}")

    # ---- summary json ----
    summary = {
        "overall_weighted": overall.to_dict(orient="records"),
        "peak_dist_all": {p: int(peak_dist[p]) for p in ["序盤", "中盤", "終盤"]},
        "peak_dist_sig": {p: int(sig_peak_dist[p]) for p in ["序盤", "中盤", "終盤"]},
        "n_sig": n_sig,
        "n_timing_dirs": int(len(t)),
        "max_range_pp": round(float(sig_dirs["attr_range"].max()) * 100, 1),
        "max_range_dir": sig_dirs.iloc[0]["directory"],
    }
    with open(RESULTS / "attrition_timing_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

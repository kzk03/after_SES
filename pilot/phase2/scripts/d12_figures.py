"""D12: 論文用の図を作る。

Fig 1: 月次推移 — 人間のレビュー件数は減っていないのに人間比率だけが下がる(dilution)
Fig 2: イベントスタディ — AI 導入後、AI 側は増えるが人間側は動かない
Fig 3: レビュー種別 — AI は COMMENTED しか出さない(判断していない)
Fig 7: 委譲の乖離 — AI の検出は伸びるが、手渡せる形で出す割合は落ちている

図中のラベルは論文投稿を想定して英語にする。

実行: uv run python pilot/phase2/scripts/d12_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# outputs_full の図を作るときは、生データ側も全期間版を読む(fig1 が build_count_panel を呼ぶため)
if os.environ.get("PHASE2_OUT") == "outputs_full":
    os.environ.setdefault("PILOT_RAW", "raw_full")
    os.environ.setdefault("PILOT_INTERIM", "interim_full")
    os.environ.setdefault("PILOT_PROCESSED", "processed_full")

from common import OUT_DIR, PHASE2_DIR, build_count_panel

# 全期間データで回すときは PHASE2_FIG=figures_full として既存の図を上書きしないようにする
FIG_DIR = PHASE2_DIR / os.environ.get("PHASE2_FIG", "figures")
plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# Okabe-Ito の青と朱(色覚多様性に安全な組)。CI/legacy bot は中立のグレー
HUMAN_C = "#0072B2"
AI_C = "#D55E00"
GREY = "#7F7F7F"
INK2 = "#444444"


def _human_verdict_rate() -> float:
    """D10 の state 分布から人間の判断率(APPROVED + CHANGES_REQUESTED の割合)を計算する。"""
    sd = json.loads((OUT_DIR / "d10_review_kind.json").read_text())["state_distribution"]["human"]
    return (sd.get("APPROVED", 0) + sd.get("CHANGES_REQUESTED", 0)) / sum(sd.values())


def fig1_trend() -> None:
    """列幅(\\columnwidth)の図。人間系列は論文の主指標である「PR 作成者を除いた人間レビュー」。

    (b) の H_E は論文本文と同じ定義(作成者を含む全人間 / 全レビュー)のまま。
    """
    m = pd.read_csv(OUT_DIR / "d1_monthly_trend.csv")
    panel = build_count_panel()
    na = panel.groupby("month")["n_E_human_nonauthor"].sum()
    m["n_E_human_nonauthor"] = m["month"].map(na)
    # 最終月は収集時点で途中までしか埋まっていないので図からは落とす
    m = m.iloc[:-1].reset_index(drop=True)
    x = np.arange(len(m))
    tick_idx = [i for i, mo in enumerate(m["month"]) if mo.endswith("-01")]

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.3), sharex=True)
    ax = axes[0]
    h = m["n_E_human_nonauthor"] / 1000
    a = m["n_E_agent"] / 1000
    ax.plot(x, h, "-", color=HUMAN_C, lw=1.8, label="Non-author human reviews")
    ax.plot(x, a, "-", color=AI_C, lw=1.8, label="AI reviews")
    ax.set_ylabel("Review events per month\n(thousands)")
    ax.set_ylim(0, 44)
    ax.grid(axis="y", color="#e3e3e3", lw=0.6)
    ax.set_axisbelow(True)
    # 左下(AI がほぼ 0、人間系列より下)が空いているのでそこに置く
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.06), fontsize=8, handlelength=1.6)
    ax.set_title("(a) Volume", fontsize=9, loc="left")
    # 注釈はデータから計算する(期間を変えても正しい値が出るように)
    h0, h1 = h.iloc[0], h.iloc[-1]
    a1 = a.iloc[-1]
    ax.annotate(f"{(h1 / h0 - 1) * 100:+.1f}% since {m['month'].iloc[0]}", xy=(x[-1], h1),
                xytext=(x[-1], 37.5), fontsize=8, color=INK2, ha="right")
    ax.annotate(f"{a1:.1f}k", xy=(x[-1], a1), xytext=(x[-1], a1 + 2.2),
                fontsize=8, color=INK2, ha="right")

    ax = axes[1]
    he = m["H_E_aggregate"]
    ax.plot(x, he, "-", color="black", lw=1.8)
    ax.set_ylabel("$H_E$ = human / all\nreview events")
    ax.set_ylim(0.70, 1.02)
    ax.grid(axis="y", color="#e3e3e3", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("(b) Human share $H_E$ (PR authors included)", fontsize=9, loc="left")
    ax.annotate(f"{he.iloc[0]:.2f}", xy=(0, he.iloc[0]), xytext=(0.3, he.iloc[0] - 0.035),
                fontsize=8, color=INK2)
    ax.annotate(f"{he.iloc[-1]:.3f}", xy=(x[-1], he.iloc[-1]), xytext=(x[-1], he.iloc[-1] - 0.04),
                fontsize=8, color=INK2, ha="right")
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(m["month"].iloc[tick_idx], rotation=45, ha="right")
    ax.set_xlim(-0.8, len(m) - 0.2)

    fig.tight_layout(h_pad=0.8)
    fig.savefig(FIG_DIR / "fig1_dilution_trend.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_dilution_trend.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig1: non-author {m['n_E_human_nonauthor'].iloc[0]:.0f} -> {m['n_E_human_nonauthor'].iloc[-1]:.0f}"
          f" ({(h1 / h0 - 1) * 100:+.1f}%), AI last {m['n_E_agent'].iloc[-1]:.0f},"
          f" H_E {he.iloc[0]:.3f} -> {he.iloc[-1]:.5f}")


def _event_series(models: dict, y: str) -> tuple[list[int], list[float], list[float]]:
    r = models[y]
    ks, betas, ses = [], [], []
    for j in [-3, -2, -1, 0, 1, 2, 3]:
        if j == -1:
            ks.append(j); betas.append(0.0); ses.append(0.0)
            continue
        name = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
        if name not in r["params"]:
            continue
        ks.append(j); betas.append(r["params"][name]); ses.append(r["se_repo"][name])
    return ks, betas, ses


def fig2_event_study() -> None:
    """\\textwidth の図。(b) は論文の主指標 log(1 + 作成者以外の人間レビュー件数)。"""
    d4 = json.loads((OUT_DIR / "d4_repo_event_study.json").read_text())["models"]
    d7 = json.loads((OUT_DIR / "d7_burden_event_study.json").read_text())["models"]

    # 結果変数はタイトルに書き、縦軸は係数の単位だけにする(ラベルの詰まりを避ける)
    panels = [
        (d4, "log_n_E_agent", "(a) log(1 + AI reviews)", AI_C, "Estimate (log points)"),
        (d4, "log_n_E_human_nonauthor", "(b) log(1 + non-author\n     human reviews)", HUMAN_C,
         "Estimate (log points)"),
        (d4, "H_E", "(c) Human share $H_E$", "black", "Estimate (share)"),
        (d7, "log_ai_before_approval", "(d) AI reviews before\n     first human approval", AI_C,
         "Estimate (log points)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.35))
    for ax, (models, y, title, color, ylab) in zip(axes, panels):
        ks, betas, ses = _event_series(models, y)
        lo = [b - 1.96 * s for b, s in zip(betas, ses)]
        hi = [b + 1.96 * s for b, s in zip(betas, ses)]
        ax.axhline(0, color=GREY, lw=0.8, ls="-")
        ax.axvline(-0.5, color=GREY, lw=0.8, ls="--")
        ax.errorbar(ks, betas, yerr=[np.array(betas) - np.array(lo), np.array(hi) - np.array(betas)],
                    fmt="o", color=color, ms=3.5, capsize=2, lw=1.2)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("Months since adoption", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax.set_xticklabels(["−3", "−2", "−1", "0", "1", "2", "3"])
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="#e3e3e3", lw=0.6)
        ax.set_axisbelow(True)
        print(f"  fig2 {y}: k+3 = {betas[-1]:+.4f} (se {ses[-1]:.4f})")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG_DIR / "fig2_event_study.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_event_study.png", bbox_inches="tight")
    plt.close(fig)


def fig3_review_kind() -> None:
    """列幅(\\columnwidth)の単一パネル: AI エージェント各アカウントの判断率。

    旧版の (a)(人間 vs AI の state 構成)は d10(モジュール対応パネル)由来で、本文の
    全件値(d45: 309 + 9 / 143,971 = 0.22%)と食い違ううえ、本文 1 文で言えるので落とした。
    アカウント別の値も本文と同じ d45(全 human/bot-authored PR、重複除去後)から取る。
    これで本文の 18 アカウント・Copilot 78,697 件と図が一致する(旧版は d11 の 78,700)。
    """
    d45 = json.loads((OUT_DIR / "d45_verdict_boundary.json").read_text())
    ag = pd.DataFrame([{"login": k, "n": v["reviews"], "verdict": v["APPROVED"] + v["CHANGES_REQUESTED"]}
                       for k, v in d45["agents"].items()])
    ag["rate"] = ag["verdict"] / ag["n"] * 100
    ag = ag.sort_values("n", ascending=True).reset_index(drop=True)
    ci = d45["ci_bots_total"]
    lr = (ci["APPROVED"] + ci["CHANGES_REQUESTED"]) / ci["reviews"] * 100

    fig, ax = plt.subplots(figsize=(3.4, 3.05))
    y = np.arange(len(ag))
    ax.barh(y, ag["rate"], color=AI_C, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(ag["login"], fontsize=7)
    ax.tick_params(axis="y", length=0)
    # 値ラベルで棒の右側の空白を埋める(0 の棒も「0」と明示する)
    for yi, r in zip(y, ag.itertuples()):
        val = "0" if r.verdict == 0 else (f"{r.rate:.1f}%" if r.rate >= 1 else f"{r.rate:.2f}%")
        ax.text(r.rate + 1.2, yi, f"{val}  (n={r.n:,})", va="center", ha="left", fontsize=7, color=INK2)
    ax.axvline(lr, color=GREY, ls="--", lw=1.1)
    ax.text(lr - 1.2, 1.5, f"CI/legacy bots\n(pooled): {lr:.1f}%", fontsize=7,
            color=INK2, ha="right", va="center")
    ax.set_xlim(0, 70)
    ax.set_ylim(-0.6, len(ag) - 0.4)
    ax.set_xlabel("Verdict rate (%)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_review_kind.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_review_kind.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig3: {len(ag)} accounts, total {ag['n'].sum():,} reviews, {ag['verdict'].sum()} verdicts;"
          f" CI/legacy pooled {lr:.2f}%;"
          f" {dict(zip(ag['login'], ag['rate'].round(3)))}")


def fig4_overlap() -> None:
    """AI と人間が同じ箇所を見ているか、そしてコメントの長さ(D21)。"""
    path = OUT_DIR / "d21_overlap.parquet"
    if not path.exists():
        print("   (d21_overlap.parquet がないので fig4 はスキップ)")
        return
    df = pd.read_parquet(path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    # (a) 行単位の重なり: AI×人間 と 人間×人間 を同じ PR で比べる
    ax = axes[0]
    sub = df.dropna(subset=["line_overlap", "human_human_overlap"])
    data = [sub["line_overlap"], sub["human_human_overlap"]]
    bp = ax.boxplot(data, labels=["AI × human", "human × human"], widths=0.5,
                    showfliers=False, patch_artist=True, medianprops={"color": "black"})
    for patch, c in zip(bp["boxes"], [AI_C, HUMAN_C]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for i, s in enumerate(data, start=1):
        ax.scatter([i], [s.mean()], marker="D", color="black", s=16, zorder=3)
    ax.set_ylabel("Jaccard overlap of commented lines")
    ax.set_title("(a) AI comments on different lines\n    than humans do", fontsize=8.5, loc="left")
    ax.text(0.5, 0.04, f"n={len(sub):,} PRs", transform=ax.transAxes, fontsize=7.5,
            ha="center", color="#555555")

    # (b) コメント1件あたりの長さ
    ax = axes[1]
    ai_len = df["ai_chars_visible"].sum() / max(df["ai_comments"].sum(), 1)
    hu_len = df["human_chars_visible"].sum() / max(df["human_comments"].sum(), 1)
    bars = ax.bar(["AI", "Human"], [ai_len, hu_len], color=[AI_C, HUMAN_C], width=0.5, alpha=0.85)
    for b, v in zip(bars, [ai_len, hu_len]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Characters per review comment")
    ax.set_title(f"(b) One AI comment = {ai_len / max(hu_len, 1):.1f} human comments\n    of reading",
                 fontsize=8.5, loc="left")
    ax.set_ylim(0, ai_len * 1.25)

    fig.suptitle("AI reviews are complementary in location but far longer to read", fontsize=9.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_overlap.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_overlap.png", bbox_inches="tight")
    plt.close(fig)


def fig5_verdict_circuit() -> None:
    """人間が作成した PR の最初の承認者(D45、全 human-authored PR、重複除去・分類修正後)。

    本文(06_rq3)と同じ母集団にするため d45 の exceptions_all_corrected を使う。
    3 値の比が 3000 倍あり線形軸では機械側の棒が見えないので、対数軸のドットプロットにする
    (対数軸で棒の長さは意味を持たないので棒にはしない)。本文 1 文で足りるので、論文では
    図を落として文だけにする選択肢もある。
    """
    p45 = OUT_DIR / "d45_verdict_boundary.json"
    if not p45.exists():
        print("   (d45 の出力がないので fig5 はスキップ)")
        return
    ex = json.loads(p45.read_text())["exceptions_all_corrected"]
    fa = ex["first_approver_cat"]
    total = sum(fa.values())
    assert total == ex["n_with_any_approval"], (total, ex["n_with_any_approval"])

    order = [("human", "Human"), ("agent", "AI agent"), ("legacy_bot", "CI/legacy bot")]
    colors = [HUMAN_C, AI_C, GREY]
    fig, ax = plt.subplots(figsize=(3.4, 1.35))
    ys = np.arange(len(order))[::-1]
    for y, (k, _), c in zip(ys, order, colors):
        n = fa.get(k, 0)
        v = n / total * 100
        ax.hlines(y, 10, n, color=c, lw=1.0, alpha=0.5)
        ax.plot(n, y, "o", color=c, ms=5.5)
        pct = f"{v:.2f}%" if v >= 1 else f"{v:.3f}%"
        if n > 1e5:  # 人間は右端なので左側に置く
            ax.text(n / 1.5, y + 0.33, f"{n:,} ({pct})", va="center", ha="right", fontsize=7.5, color=INK2)
        else:
            ax.text(n * 1.5, y, f"{n:,} ({pct})", va="center", ha="left", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(10, 2e6)
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in order], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="#e3e3e3", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"First approvals (log scale; n = {total:,} PRs)", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_verdict_circuit.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_verdict_circuit.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig5: first approver {fa} of {total}")


def fig6_role_shift() -> None:
    """AI 側の役割は静的でない: 判断率の上昇とエージェントの世代交代(D28)。"""
    pr_path = OUT_DIR / "d5_pr_level.parquet"
    tri_path = OUT_DIR / "d26_triage.parquet"
    if not (pr_path.exists() and tri_path.exists()):
        print("   (d5/d26 の出力がないので fig6 はスキップ)")
        return
    pr = pd.read_parquet(pr_path)
    m = pr.groupby("month").agg(
        ai_appr=("n_ai_approved", "sum"), ai_cr=("n_ai_changes_requested", "sum"),
        ai_com=("n_ai_commented", "sum")).reset_index()
    m["total"] = m["ai_appr"] + m["ai_cr"] + m["ai_com"]
    m = m[m["total"] >= 50].reset_index(drop=True)
    m = m.iloc[:-1].reset_index(drop=True)  # 最終月は収集途中で不完全
    m["verdict_rate"] = (m["ai_appr"] + m["ai_cr"]) / m["total"]

    tri = pd.read_parquet(tri_path)
    tri["month"] = tri["created_at"].str[:7]
    sh = tri.groupby(["month", "agent"]).size().rename("n").reset_index()
    sh["share"] = sh["n"] / sh.groupby("month")["n"].transform("sum")
    piv = sh.pivot(index="month", columns="agent", values="share").fillna(0)
    piv = piv[piv.index.isin(m["month"])]
    top = piv.sum().sort_values(ascending=False).index[:4]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))

    # (a) AI の判断率の推移。人間は 42.1% で桁が違うので軸には載せず注記で示す
    ax = axes[0]
    x = range(len(m))
    ax.plot(x, m["verdict_rate"] * 100, "o-", color=AI_C, ms=3.5, lw=1.6)
    ax.set_ylim(0, max(m["verdict_rate"]) * 100 * 1.45)
    ax.set_ylabel("Verdict rate of LLM agents (%)")
    ax.set_xticks(list(x)[::4])
    ax.set_xticklabels(m["month"][::4], rotation=45, ha="right")
    hu = _human_verdict_rate()
    tr = json.loads((OUT_DIR / "d28_ai_role_shift.json").read_text())["verdict_rate_trend"]
    ratio = hu / max(float(m["verdict_rate"].tail(6).mean()), 1e-9)
    ax.set_title(f"(a) Rising, but still ~{ratio:.0f}x below humans", fontsize=8.5, loc="left")
    ax.text(0.03, 0.92, f"Human level: {hu:.1%}  (off scale)",
            transform=ax.transAxes, fontsize=7.5, color=HUMAN_C)
    ax.text(0.03, 0.82, f"trend {tr['slope'] * 1200:+.3f} pt/yr (p={tr['p']:.3f})",
            transform=ax.transAxes, fontsize=7.5, color="#555555")

    # (b) エージェントのシェア推移(claude = 判断を出す唯一のエージェント)
    ax = axes[1]
    xs = range(len(piv))
    palette = {"claude": "#9467bd"}
    bottom = np.zeros(len(piv))
    for i, c in enumerate(top):
        col = palette.get(c, ["#c7c7c7", "#8c8c8c", "#5f5f5f", "#3a3a3a"][i % 4])
        lab = c if c != "copilot-pull-request-reviewer" else "copilot-pr-reviewer"
        ax.bar(xs, piv[c].values * 100, bottom=bottom, color=col, width=0.85,
               label=lab, edgecolor="none")
        bottom += piv[c].values * 100
    ax.set_ylabel("Share of AI review comments (%)")
    ax.set_xticks(list(xs)[::4])
    ax.set_xticklabels(piv.index[::4], rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("(b) claude (the only agent that judges) is growing", fontsize=8.5, loc="left")

    fig.suptitle("The boundary is not static: it moves with agent design, not capability",
                 fontsize=9.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_role_shift.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_role_shift.png", bbox_inches="tight")
    plt.close(fig)


def fig7_delegation_gap() -> None:
    """検出と委譲可能量の乖離。論文の処方の根拠になる図(列幅 \\columnwidth、3.4in)。

    2系列とも『人間コメントに占める割合』で単位が同じなので単一軸にする
    (二軸は読み手を誤らせるので使わない)。差そのものが主張なので、間を塗る。
    最終月(2026-09)は収集途中なので、白抜きマーカーと点線でつなぎ、凡例で断る。
    """
    # 作成者本人の返信を除いた人間コメントを分母にする(D44)。
    # 旧版は d41_monthly_ceiling.csv を読んでいたが、どのスクリプトも書いておらず古い値のままだった。
    path = OUT_DIR / "d44_author_excluded.json"
    if not path.exists():
        print("  [skip] fig7: d44_author_excluded.json がない")
        return
    ser = json.loads(path.read_text())["monthly"]["nonauthor"]["series"]
    m = pd.DataFrame([{"month": k, "detected": v["det"], "delegable": v["dl"]}
                      for k, v in sorted(ser.items())])
    x = np.arange(len(m))
    det = m["detected"].to_numpy() * 100
    del_ = m["delegable"].to_numpy() * 100
    PARTIAL = "2026-09"
    has_partial = m["month"].iloc[-1] == PARTIAL
    nf = len(m) - 1 if has_partial else len(m)  # 完全な月の数

    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    ax.fill_between(x, del_, det, color=GREY, alpha=0.14, linewidth=0)
    for s, c, lab in [(det, HUMAN_C, "AI flagged the same location (detected)"),
                      (del_, AI_C, "…with an applicable patch (delegable)")]:
        ax.plot(x[:nf], s[:nf], color=c, lw=1.6, label=lab)
        if has_partial:
            ax.plot(x[nf - 1:], s[nf - 1:], color=c, lw=1.2, ls=":")
            ax.plot(x[-1], s[-1], "o", ms=4, mfc="white", mec=c, mew=1.2)
    if has_partial:
        ax.plot([], [], "o", ms=4, mfc="white", mec=INK2, mew=1.2, label=f"{PARTIAL}: partial month")

    # 崩壊した月(Copilot が suggestion 形式をほぼやめた月)。文字は線に重ならないよう 2 系列の間の帯に置く
    if "2026-05" in set(m["month"]):
        i = int(m.index[m["month"] == "2026-05"][0])
        ax.annotate("Copilot stops\nsuggestion blocks", xy=(i, del_[i]), xytext=(i + 2.6, 13.6),
                    fontsize=7, color=INK2, ha="right", va="center",
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8,
                                    shrinkA=1, shrinkB=2.5, relpos=(0.72, 0.0), connectionstyle="arc3,rad=0"))

    tick_idx = [j for j, mo in enumerate(m["month"]) if mo[5:] in ("02", "05", "08", "11")]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(m["month"].iloc[tick_idx], rotation=45, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("% of non-author reviewer\ninline comments", fontsize=7.5)
    ax.set_ylim(0, 40)
    ax.set_xlim(-0.5, len(m) - 0.5)
    ax.grid(axis="y", color="#e3e3e3", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.6, borderaxespad=0.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig7_delegation_gap.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig7_delegation_gap.pdf  (2026-04 det {det[m['month'] == '2026-04'][0]:.2f}% /"
          f" dl {del_[m['month'] == '2026-04'][0]:.2f}%; 2026-05 dl {del_[m['month'] == '2026-05'][0]:.2f}%;"
          f" last {m['month'].iloc[-1]} det {det[-1]:.2f}% / dl {del_[-1]:.2f}%)")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_trend()
    fig2_event_study()
    fig3_review_kind()
    fig4_overlap()
    fig5_verdict_circuit()
    fig6_role_shift()
    fig7_delegation_gap()
    print(f"-> {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"   {p.name}")


if __name__ == "__main__":
    main()

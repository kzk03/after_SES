"""D12: 論文用の図を作る。

Fig 1: 月次推移 — 人間のレビュー件数は減っていないのに人間比率だけが下がる(dilution)
Fig 2: イベントスタディ — AI 導入後、AI 側は増えるが人間側は動かない
Fig 3: レビュー種別 — AI は COMMENTED しか出さない(判断していない)

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

from common import OUT_DIR, PHASE2_DIR

# 全期間データで回すときは PHASE2_FIG=figures_full として既存の図を上書きしないようにする
FIG_DIR = PHASE2_DIR / os.environ.get("PHASE2_FIG", "figures")
plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

HUMAN_C = "#1f77b4"
AI_C = "#d62728"
GREY = "#888888"


def fig1_trend() -> None:
    m = pd.read_csv(OUT_DIR / "d1_monthly_trend.csv")
    # 最終月は収集時点で途中までしか埋まっていないので図からは落とす
    m = m.iloc[:-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    ax = axes[0]
    x = range(len(m))
    ax.plot(x, m["n_E_human"] / 1000, "o-", color=HUMAN_C, label="Human reviews", ms=3.5, lw=1.6)
    ax.plot(x, m["n_E_agent"] / 1000, "s-", color=AI_C, label="AI reviews", ms=3.5, lw=1.6)
    ax.set_ylabel("Review events (thousands)")
    ax.set_xticks(list(x)[::3])
    ax.set_xticklabels(m["month"][::3], rotation=45, ha="right")
    ax.legend(frameon=False, loc="center left")
    ax.set_title("(a) Absolute volume", fontsize=9, loc="left")
    ax.set_ylim(bottom=0)
    # 注釈はデータから計算する(期間を変えても正しい値が出るように)
    h0, h1 = m["n_E_human"].iloc[0], m["n_E_human"].iloc[-1]
    a1 = m["n_E_agent"].iloc[-1]
    top = max(m["n_E_human"].max(), m["n_E_agent"].max()) / 1000
    ax.annotate(f"{(h1 / h0 - 1) * 100:+.0f}%", xy=(len(m) - 1, h1 / 1000),
                xytext=(len(m) - 3.4, top * 1.02), fontsize=8, color=HUMAN_C)
    ax.annotate(f"{a1 / 1000:.0f}k", xy=(len(m) - 1, a1 / 1000),
                xytext=(len(m) - 2.6, a1 / 1000 + top * 0.06), fontsize=8, color=AI_C)

    ax = axes[1]
    ax.plot(x, m["H_E_aggregate"], "o-", color="black", ms=3.5, lw=1.6)
    ax.set_ylabel("Human share of review events ($H_E$)")
    ax.set_xticks(list(x)[::3])
    ax.set_xticklabels(m["month"][::3], rotation=45, ha="right")
    ax.set_ylim(0.75, 1.0)
    ax.set_title("(b) Human share", fontsize=9, loc="left")
    ax.annotate(f"{m['H_E_aggregate'].iloc[0]:.2f}", xy=(0, m["H_E_aggregate"].iloc[0]),
                xytext=(0.5, 0.955), fontsize=8)
    ax.annotate(f"{m['H_E_aggregate'].iloc[-1]:.2f}", xy=(len(m) - 1, m["H_E_aggregate"].iloc[-1]),
                xytext=(len(m) - 2.6, 0.774), fontsize=8)

    fig.suptitle("Human review volume grows while the human share falls", fontsize=9.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_dilution_trend.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_dilution_trend.png", bbox_inches="tight")
    plt.close(fig)


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
    d4 = json.loads((OUT_DIR / "d4_repo_event_study.json").read_text())["models"]
    d7 = json.loads((OUT_DIR / "d7_burden_event_study.json").read_text())["models"]

    panels = [
        (d4, "log_n_E_agent", "(a) AI review volume", AI_C, "log(1+n)"),
        (d4, "log_n_E_human", "(b) Human review volume", HUMAN_C, "log(1+n)"),
        (d4, "H_E", "(c) Human share $H_E$", "black", "share"),
        (d7, "log_ai_before_approval", "(d) AI comments read\nbefore human approval", AI_C, "log(1+n)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.6))
    for ax, (models, y, title, color, ylab) in zip(axes, panels):
        ks, betas, ses = _event_series(models, y)
        lo = [b - 1.96 * s for b, s in zip(betas, ses)]
        hi = [b + 1.96 * s for b, s in zip(betas, ses)]
        ax.axhline(0, color=GREY, lw=0.8, ls="-")
        ax.axvline(-0.5, color=GREY, lw=0.8, ls="--")
        ax.errorbar(ks, betas, yerr=[np.array(betas) - np.array(lo), np.array(hi) - np.array(betas)],
                    fmt="o", color=color, ms=4, capsize=2.5, lw=1.3)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("Months since AI review adoption")
        ax.set_ylabel(ylab)
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    fig.suptitle("AI review volume rises and the human share falls, but human review volume does not decline",
                 fontsize=9.5, y=1.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_event_study.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_event_study.png", bbox_inches="tight")
    plt.close(fig)


def fig3_review_kind() -> None:
    d = json.loads((OUT_DIR / "d10_review_kind.json").read_text())["state_distribution"]
    prof = pd.read_csv(OUT_DIR / "d11_agent_profile.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))

    ax = axes[0]
    cats = ["Human", "AI agent"]
    states = ["APPROVED", "CHANGES_REQUESTED", "COMMENTED"]
    colors = ["#2ca02c", "#ff7f0e", "#c7c7c7"]
    bottoms = np.zeros(2)
    totals = [sum(d["human"].values()), sum(d["agent"].values())]
    for st, c in zip(states, colors):
        vals = np.array([d["human"][st] / totals[0], d["agent"][st] / totals[1]])
        ax.bar(cats, vals, bottom=bottoms, color=c, label=st.replace("_", " ").title(), width=0.55)
        for i, v in enumerate(vals):
            if v > 0.03:
                ax.text(i, bottoms[i] + v / 2, f"{v:.1%}", ha="center", va="center", fontsize=8)
        bottoms += vals
    ax.set_ylabel("Share of review events")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_title("(a) AI almost never issues a verdict", fontsize=8.5, loc="left")

    ax = axes[1]
    agents = prof[(prof["category"] == "agent") & (prof["n_reviews"] >= 5)]
    agents = agents.sort_values("n_reviews", ascending=True)
    labels = [f"{r.login}\n(n={r.n_reviews:,})" for r in agents.itertuples()]
    legacy_rate = prof[prof["category"] == "legacy_bot"]
    lr = (legacy_rate["APPROVED"].sum() + legacy_rate["CHANGES_REQUESTED"].sum()) / legacy_rate["n_reviews"].sum()
    ax.barh(labels, agents["judgment_share"] * 100, color=AI_C, height=0.6)
    ax.axvline(lr * 100, color=GREY, ls="--", lw=1.2)
    ax.text(lr * 100 - 1.2, -0.4, f"legacy bots (CI etc.): {lr:.0%}",
            fontsize=7.5, color="#555555", rotation=90, va="bottom", ha="right")
    ax.set_xlim(0, 70)
    ax.set_xlabel("Verdict rate: % of reviews that are\nAPPROVED or CHANGES_REQUESTED")
    ax.set_title("(b) Deterministic bots judge; LLM agents do not", fontsize=8.5, loc="left")
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_review_kind.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_review_kind.png", bbox_inches="tight")
    plt.close(fig)


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
    """★核の図: 判断の回路は機械全体に対して閉じている(D24, D11)。"""
    p24 = OUT_DIR / "d24_approval_substitution.json"
    prof_path = OUT_DIR / "d11_agent_profile.csv"
    if not (p24.exists() and prof_path.exists()):
        print("   (d24/d11 の出力がないので fig5 はスキップ)")
        return
    d24 = json.loads(p24.read_text())
    prof = pd.read_csv(prof_path)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))

    # (a) 人間が作成した PR の「最初の承認者」は誰か
    ax = axes[0]
    fa = d24.get("first_approver_human_authored", {})
    total = sum(fa.values()) or 1
    order = [("human", "Human"), ("agent", "AI agent"), ("legacy_bot", "CI bot")]
    vals = [fa.get(k, 0) / total for k, _ in order]
    labels = [f"{lab}\n({fa.get(k, 0):,})" for k, lab in order]
    colors = [HUMAN_C, AI_C, GREY]
    bars = ax.barh(labels[::-1], [v * 100 for v in vals][::-1], color=colors[::-1], height=0.6)
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2,
                f"{v:.1%}" if v >= 0.001 else f"{v:.2%}", va="center", fontsize=8)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Share of first approvals (%)")
    ax.set_title("(a) Who approves human-authored PRs first?", fontsize=8.5, loc="left")
    ax.tick_params(axis="y", labelsize=8)

    # (b) 判断率: 人間 / 決定論的ボット / LLM エージェント
    ax = axes[1]
    ag = prof[prof["category"] == "agent"]
    lg = prof[prof["category"] == "legacy_bot"]
    ag_rate = (ag["APPROVED"].sum() + ag["CHANGES_REQUESTED"].sum()) / ag["n_reviews"].sum()
    lg_rate = (lg["APPROVED"].sum() + lg["CHANGES_REQUESTED"].sum()) / lg["n_reviews"].sum()
    hu_rate = 0.421  # D10: (601,476 + 50,612) / 1,548,762
    names = ["Human", "Deterministic\nbots (CI)", "LLM agents"]
    vals = [hu_rate, lg_rate, ag_rate]
    bars = ax.bar(names, [v * 100 for v in vals], color=[HUMAN_C, GREY, AI_C], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.2,
                f"{v:.1%}" if v >= 0.01 else f"{v:.2%}", ha="center", fontsize=8)
    ax.set_ylabel("Verdict rate (%)\nAPPROVED or CHANGES_REQUESTED")
    ax.set_ylim(0, 72)
    ax.set_title("(b) LLM agents almost never issue a verdict", fontsize=8.5, loc="left")

    fig.suptitle("The verdict circuit is closed to machines in general, not just to AI",
                 fontsize=9.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_verdict_circuit.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_verdict_circuit.png", bbox_inches="tight")
    plt.close(fig)


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
    ax.set_title("(a) Rising, but still ~60x below humans", fontsize=8.5, loc="left")
    ax.text(0.03, 0.92, "Human level: 42.1%  (off scale)",
            transform=ax.transAxes, fontsize=7.5, color=HUMAN_C)
    ax.text(0.03, 0.82, f"trend +0.038 pt/yr (p=0.012)",
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


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_trend()
    fig2_event_study()
    fig3_review_kind()
    fig4_overlap()
    fig5_verdict_circuit()
    fig6_role_shift()
    print(f"-> {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"   {p.name}")


if __name__ == "__main__":
    main()

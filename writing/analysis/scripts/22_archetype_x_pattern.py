# -*- coding: utf-8 -*-
"""開発者archetype × ラベル系列パターン交差表 (§4 横串の収束を定量化).

§2.4 (script 14) で C Late starter / D Returner で IRL > RF だと示し、
§2.5 (paper Table 2) で A2 (レビュー専念型) 全 4 区間で IRL > RF だと示し、
§2.2 (script 09) で IRL Lasso 残差プロファイル = 「不定期だが粘り強い」と示した。

本スクリプトは「A2 開発者が C Late starter / D Returner に偏在するか」を
直接の交差表 + χ² 検定で確認する。これが横串の収束を 1 つのテーブルで締める
ピースとなる。

集約単位:
  - 1 行 = 1 ペア (dev, dir)。各ペアは 4 区間ラベルで 16 通りの pattern を持つ
  - pattern は §2.4 と同じ 7 クラス (A Steady, B Dropout, C Late starter,
    D Returner, E Sporadic, F Single-pulse, Z Inactive) に集約
  - dev archetype は pair_trajectories.csv の `archetype` 列 (A1-A4) を使用
  - 母集団 = bot 除外後 + archetype 既知のペア

出力:
  results/archetype_x_pattern.csv
    rows = dev archetype × pattern class
    cols = n_obs / share_in_archetype / expected_under_independence / lift
  results/archetype_x_pattern_chi2.json
    全体の χ² 統計と p 値
  results/fig_archetype_x_pattern.pdf
    左: 観測比率の積み上げ棒、右: lift ヒートマップ
"""
from __future__ import annotations

import json
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
REPO_ROOT = ROOT.parents[1]
BOT_CSV = REPO_ROOT / "outputs/analysis/bot_detection/bot_candidates_main32.csv"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]

PATTERN_TO_CLASS = {
    "1111": ("A", "A. Steady"),
    "1000": ("B", "B. Dropout"),
    "1100": ("B", "B. Dropout"),
    "1110": ("B", "B. Dropout"),
    "0001": ("C", "C. Late starter"),
    "0011": ("C", "C. Late starter"),
    "0111": ("C", "C. Late starter"),
    "1001": ("D", "D. Returner"),
    "1011": ("D", "D. Returner"),
    "1101": ("D", "D. Returner"),
    "1010": ("E", "E. Sporadic"),
    "0101": ("E", "E. Sporadic"),
    "0010": ("F", "F. Single-pulse"),
    "0100": ("F", "F. Single-pulse"),
    "0110": ("F", "F. Single-pulse"),
    "0000": ("Z", "Z. Inactive"),
}
CLASS_ORDER = ["A", "B", "C", "D", "E", "F", "Z"]
CLASS_JP = {
    "A": "A. Steady (1111)",
    "B": "B. Dropout",
    "C": "C. Late starter",
    "D": "D. Returner",
    "E": "E. Sporadic",
    "F": "F. Single-pulse",
    "Z": "Z. Inactive (0000)",
}
ARCH_ORDER = ["A1", "A2", "A3", "A4"]
ARCH_JP = {
    "A1": "A1 低活動の新人",
    "A2": "A2 レビュー専念型",
    "A3": "A3 高承諾のベテラン",
    "A4": "A4 常時活動型",
}


def build_pattern(df: pd.DataFrame) -> pd.Series:
    bits = [df[f"label_{h}"].astype(int).astype(str) for h in HORIZONS]
    pat = bits[0]
    for b in bits[1:]:
        pat = pat + b
    return pat


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)

    pair = pd.read_csv(DATA / "pair_trajectories.csv")
    print(f"raw pairs: {len(pair):,}")

    if BOT_CSV.exists():
        bot_df = pd.read_csv(BOT_CSV)
        bots = set(bot_df.loc[bot_df["is_bot"] == True, "email"].tolist())  # noqa: E712
        n0 = len(pair)
        pair = pair[~pair["developer"].isin(bots)].copy()
        print(f"bot exclusion: {n0:,} → {len(pair):,}  (-{n0 - len(pair):,})")

    pair = pair.dropna(subset=["archetype"]).copy()
    print(f"archetype 既知のペア: {len(pair):,}")

    pair["pattern"] = build_pattern(pair)
    pair["class"] = pair["pattern"].map(lambda p: PATTERN_TO_CLASS.get(p, ("?", "?"))[0])
    pair["class_jp"] = pair["class"].map(CLASS_JP)

    # ── 交差表 (count) ──────────────────────────────────
    ct = pd.crosstab(pair["archetype"], pair["class"])
    ct = ct.reindex(index=ARCH_ORDER, columns=CLASS_ORDER, fill_value=0)
    print("\n=== 観測度数 (dev archetype × pattern class) ===")
    print(ct.to_string())

    # ── 期待度数 (独立性下の積) ───────────────────────
    row_tot = ct.sum(axis=1)
    col_tot = ct.sum(axis=0)
    n_total = ct.values.sum()
    expected = np.outer(row_tot, col_tot) / n_total
    expected_df = pd.DataFrame(expected, index=ARCH_ORDER, columns=CLASS_ORDER)

    # ── lift = obs / expected ──────────────────────────
    lift = ct.values / expected
    lift_df = pd.DataFrame(lift, index=ARCH_ORDER, columns=CLASS_ORDER)
    print("\n=== Lift = obs / expected (>1 = 偏在) ===")
    print(lift_df.round(2).to_string())

    # ── archetype 内シェア (行方向 normalize) ─────────
    share = ct.div(row_tot, axis=0)
    print("\n=== archetype 内シェア (行方向) ===")
    print(share.round(3).to_string())

    # ── χ² 検定 (mixed パターン 5 クラスのみで) ─────
    mixed_cols = ["B", "C", "D", "E", "F"]
    ct_mixed = ct[mixed_cols]
    chi2_stat, p_val, dof, expected_mixed = chi2_contingency(ct_mixed.values)
    print(f"\n=== χ² 検定 (混合パターン B-F、A/Z は除外) ===")
    print(f"  χ² = {chi2_stat:.2f}, dof = {dof}, p = {p_val:.3e}")
    print(f"  n = {ct_mixed.values.sum()}")

    # 各クラスでの archetype 偏在 (Z スコア風)
    print("\n=== 各 pattern class での archetype 偏在 (lift, mixed のみ) ===")
    for cls in mixed_cols:
        print(f"  {CLASS_JP[cls]}:")
        for arch in ARCH_ORDER:
            l = lift_df.loc[arch, cls]
            obs = int(ct.loc[arch, cls])
            exp = expected_df.loc[arch, cls]
            arrow = "↑" if l > 1.1 else ("↓" if l < 0.9 else "≈")
            print(f"    {arch}: obs={obs:4d}, exp={exp:.1f}, lift={l:.2f} {arrow}")

    # ── 長形式 CSV 出力 ─────────────────────────────────
    long_rows = []
    for arch in ARCH_ORDER:
        for cls in CLASS_ORDER:
            long_rows.append({
                "dev_archetype": arch,
                "dev_archetype_jp": ARCH_JP[arch],
                "pattern_class": cls,
                "pattern_class_jp": CLASS_JP[cls],
                "n_obs": int(ct.loc[arch, cls]),
                "share_in_archetype": float(share.loc[arch, cls]),
                "expected_under_independence": float(expected_df.loc[arch, cls]),
                "lift": float(lift_df.loc[arch, cls]),
            })
    long_df = pd.DataFrame(long_rows)
    out_csv = OUT / "archetype_x_pattern.csv"
    long_df.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")

    chi2_summary = {
        "n_pairs_total": int(n_total),
        "n_pairs_mixed_only": int(ct_mixed.values.sum()),
        "chi2_mixed_BCDEF": float(chi2_stat),
        "dof": int(dof),
        "p_value": float(p_val),
    }
    out_json = OUT / "archetype_x_pattern_chi2.json"
    out_json.write_text(json.dumps(chi2_summary, indent=2, ensure_ascii=False))
    print(f"saved: {out_json}")

    # ── 図 (2 panel) ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # 左: 各 archetype のクラス比率 (積み上げ棒)
    ax = axes[0]
    bottom = np.zeros(len(ARCH_ORDER))
    colors = {
        "A": "#7f7f7f",  # gray (Steady)
        "B": "#d62728",  # red (Dropout)
        "C": "#1f77b4",  # blue (Late starter) — IRL 優位
        "D": "#17becf",  # cyan (Returner) — IRL 優位
        "E": "#ff7f0e",  # orange (Sporadic)
        "F": "#bcbd22",  # olive (Single-pulse)
        "Z": "#cccccc",  # light gray (Inactive)
    }
    for cls in CLASS_ORDER:
        vals = share[cls].values
        ax.bar(np.arange(len(ARCH_ORDER)), vals, bottom=bottom,
               label=CLASS_JP[cls], color=colors[cls], width=0.7,
               edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v >= 0.04:
                ax.text(i, bottom[i] + v / 2, f"{v*100:.0f}%",
                        ha="center", va="center", fontsize=8.5, color="white",
                        fontweight="bold")
        bottom += vals
    ax.set_xticks(np.arange(len(ARCH_ORDER)))
    ax.set_xticklabels([ARCH_JP[a] for a in ARCH_ORDER], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("ペア内のクラス占有率")
    ax.set_title("(a) 各開発者archetype 内の pattern class 占有率")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.10),
              ncol=4, fontsize=8.5, frameon=False)

    # 右: lift ヒートマップ (mixed のみ)
    ax = axes[1]
    lift_mixed = lift_df[mixed_cols]
    im = ax.imshow(lift_mixed.values, cmap="RdBu_r", vmin=0.5, vmax=2.0,
                   aspect="auto")
    ax.set_xticks(np.arange(len(mixed_cols)))
    ax.set_xticklabels([CLASS_JP[c] for c in mixed_cols], fontsize=9,
                       rotation=20, ha="right")
    ax.set_yticks(np.arange(len(ARCH_ORDER)))
    ax.set_yticklabels([ARCH_JP[a] for a in ARCH_ORDER], fontsize=9)
    for i, arch in enumerate(ARCH_ORDER):
        for j, cls in enumerate(mixed_cols):
            v = lift_mixed.iloc[i, j]
            obs = int(ct.loc[arch, cls])
            text_color = "white" if (v < 0.7 or v > 1.5) else "black"
            ax.text(j, i, f"{v:.2f}\nn={obs}",
                    ha="center", va="center", fontsize=8.5, color=text_color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Lift = obs / expected", fontsize=9)
    ax.set_title(f"(b) lift ヒートマップ (混合パターン B-F のみ)\n"
                 f"χ²={chi2_stat:.0f}, dof={dof}, p={p_val:.1e}")

    fig.suptitle("開発者archetype × ラベル系列パターン交差表 — "
                 "「不定期だが粘り強い」開発者像の収束確認", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_pdf = OUT / "fig_archetype_x_pattern.pdf"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_pdf}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""「tail × 学習年数」グリッド全系列の pooled AUC を統合した CSV と図を作る.

t0_compare_3series.py の後継（3系列固定 → グリッド全系列）。前提として各系列の
横断 CSV（t0_compare.py）か、IRL 未着系列は RF 単体 CSV（rf_only_blabel.py）が
生成済みであること。存在する分だけで動く（未着系列は RF のみ白抜き補完 or skip）。

出力:
  outputs/t0_grid_comparison.csv                … 全系列 long 形式
  writing/fig/fig_t0_grid.{pdf,png}             … IRL/RF/Ens 3面 × 全系列
"""
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]

# series名 → (root, 描画スタイル)。tail揃えない=inter系は暖色、tail12=align系は寒色。
SERIES = {
    "paper (buffer, 33st)":      ("outputs/t0_sweep",            dict(color="tab:gray",   marker="o", ls="-")),
    "inter_fs24 (tailHZ, 24st)": ("outputs/t0_sweep_inter_fs24", dict(color="tab:orange", marker="s", ls=":")),
    "inter_fs36 (tailHZ, 36st)": ("outputs/t0_sweep_inter_fs36", dict(color="tab:red",    marker="s", ls="-")),
    "inter_fs45 (tailHZ, 45st)": ("outputs/t0_sweep_inter_fs45", dict(color="tab:brown",  marker="s", ls="--")),
    "align_fs24 (tail12, 24st)": ("outputs/t0_sweep_align_fs24", dict(color="tab:cyan",   marker="^", ls=":")),
    "align_fs36 (tail12, 36st)": ("outputs/t0_sweep_align_fs36", dict(color="tab:blue",   marker="^", ls="-")),
}


def load_series(root: pathlib.Path) -> pd.DataFrame | None:
    """t0_comparison_pooled*.csv（サフィックス揺れ吸収）→ 無ければ rf_only のみ。"""
    hits = sorted(glob.glob(str(root / "t0_comparison_pooled*.csv")))
    if hits:
        df = pd.read_csv(hits[0])[["t0", "n", "posrate", "IRL", "RF", "Ens", "Ens_minus_best"]]
    else:
        df = pd.DataFrame(columns=["t0", "n", "posrate", "IRL", "RF", "Ens", "Ens_minus_best"])
    # IRL 未着 t0 を RF 単体集計で補完（白抜き表示用に irl_ready=False を立てる）
    rf_hits = sorted(glob.glob(str(root / "rf_only_blabel*.csv")))
    df["irl_ready"] = True
    if rf_hits:
        rf = pd.read_csv(rf_hits[0])
        rf = rf[rf["horizon"] == "pooled"][["t0", "n", "posrate", "RF_roc"]].rename(
            columns={"RF_roc": "RF"})
        rf["RF"] = rf["RF"].round(3)
        extra = rf[~rf["t0"].isin(df["t0"])].copy()
        if len(extra):
            extra["irl_ready"] = False
            df = pd.concat([df, extra], ignore_index=True)
    if df.empty:
        return None
    return df.sort_values("t0").reset_index(drop=True)


def main() -> None:
    frames = []
    for name, (root, _) in SERIES.items():
        d = load_series(ROOT / root)
        if d is None:
            print(f"[skip] {name}: 集計CSVなし ({root})")
            continue
        d.insert(0, "series", name)
        frames.append(d)
    if not frames:
        print("集計対象なし")
        return
    merged = pd.concat(frames, ignore_index=True)
    out_csv = ROOT / "outputs/t0_grid_comparison.csv"
    merged.to_csv(out_csv, index=False)

    # 見やすい pivot（IRL）をコンソールに
    for col in ["IRL", "RF", "Ens"]:
        piv = merged.pivot_table(index="t0", columns="series", values=col)
        print(f"\n=== {col} (pooled AUC, B定義) ===")
        print(piv.to_string(float_format=lambda x: f"{x:.3f}"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, col in zip(axes, ["IRL", "RF", "Ens"]):
        for name, (_, style) in SERIES.items():
            d = merged[(merged["series"] == name)].dropna(subset=[col])
            if d.empty:
                continue
            full = d[d["irl_ready"]]
            ax.plot(pd.to_datetime(full["t0"]), full[col], label=name, **style)
            part = d[~d["irl_ready"]]
            if col == "RF" and len(part):  # IRL未着点は白抜き
                ax.plot(pd.to_datetime(part["t0"]), part[col], ls="none",
                        marker=style["marker"], mfc="none", color=style["color"])
        ax.axvline(pd.Timestamp("2023-01-01"), ls="--", c="gray", alpha=.5)
        ax.set_title(col)
        ax.set_ylim(0.70, 0.86)
        ax.grid(alpha=.3)
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("pooled AUC-ROC (B-label)")
    axes[2].legend(fontsize=7, loc="lower left")
    fig.suptitle("t0 sweep grid: tail (HZ依存/12統一) × 学習年数 (24/36/45step), B-label pooled AUC")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = ROOT / f"writing/fig/fig_t0_grid.{ext}"
        fig.savefig(p, dpi=150)
        print(f"saved: {p}")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()

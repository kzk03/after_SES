# -*- coding: utf-8 -*-
"""論文(Phase B) / inter / align の3系列 pooled AUC を統合したCSVと図を作る.

前提: 各系列の横断CSVが生成済みであること。
  - outputs/t0_sweep/t0_comparison_pooled.csv            (uv run python t0_compare.py)
  - outputs/t0_sweep_inter_fs45/t0_comparison_pooled_inter.csv (INTER=1 t0_compare.py)
  - outputs/t0_sweep_align_fs24/t0_comparison_pooled_align.csv (ALIGN=1 t0_compare.py)
inter で IRL 未着の t0 は、RF単体集計 (rf_only_blabel_inter.csv) があれば
RF のみ白抜きマーカーで補完する。

出力:
  outputs/t0_sweep_inter_fs45/t0_comparison_3series.csv
  writing/fig/fig_t0_sensitivity_3series.{pdf,png}
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]

paper = pd.read_csv(ROOT / "outputs/t0_sweep/t0_comparison_pooled.csv")
inter = pd.read_csv(ROOT / "outputs/t0_sweep_inter_fs45/t0_comparison_pooled_inter.csv")
align = pd.read_csv(ROOT / "outputs/t0_sweep_align_fs24/t0_comparison_pooled_align.csv")

rf_only_path = ROOT / "outputs/t0_sweep_inter_fs45/rf_only_blabel_inter.csv"
if rf_only_path.exists():
    rf_only = pd.read_csv(rf_only_path)
    rf_only = rf_only[rf_only["horizon"] == "pooled"][["t0", "RF_roc"]].rename(
        columns={"RF_roc": "RF"})
else:
    rf_only = pd.DataFrame(columns=["t0", "RF"])

for d, name in [(paper, "paper"), (inter, "inter"), (align, "align")]:
    d["design"] = name

merged = pd.concat(
    [paper, inter, align], ignore_index=True
)[["design", "t0", "n", "posrate", "IRL", "RF", "Ens", "Ens_minus_best"]]
extra = rf_only[~rf_only["t0"].isin(inter["t0"])].copy()
if len(extra):
    extra["design"] = "inter"
    extra["RF"] = extra["RF"].round(3)
    merged = pd.concat([merged, extra], ignore_index=True)
merged = merged.sort_values(["design", "t0"]).reset_index(drop=True)
out_csv = ROOT / "outputs/t0_sweep_inter_fs45/t0_comparison_3series.csv"
merged.to_csv(out_csv, index=False)
print(merged.to_string(index=False))

styles = {
    "paper": dict(color="tab:gray", marker="o", label="paper (buffer 1y)"),
    "inter": dict(color="tab:orange", marker="s", label="inter (no buffer)"),
    "align": dict(color="tab:blue", marker="^", label="align (no buffer + tail12)"),
}
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, col in zip(axes, ["IRL", "RF", "Ens"]):
    for name, df in [("paper", paper), ("inter", inter), ("align", align)]:
        d = df.dropna(subset=[col])
        ax.plot(pd.to_datetime(d["t0"]), d[col], **styles[name])
    if col == "RF" and len(extra):
        ax.plot(pd.to_datetime(extra["t0"]), extra["RF"], ls="none",
                marker="s", mfc="none", color="tab:orange")
    ax.axvline(pd.Timestamp("2023-01-01"), ls="--", c="gray", alpha=.5)
    ax.set_title(col)
    ax.set_ylim(0.70, 0.86)
    ax.grid(alpha=.3)
    ax.tick_params(axis="x", rotation=45)
axes[0].set_ylabel("pooled AUC-ROC (B-label)")
axes[1].legend(loc="lower center", fontsize=9)
fig.suptitle("t0 sweep: design comparison (paper / inter / align), B-label pooled AUC")
fig.tight_layout()
for ext in ["pdf", "png"]:
    p = ROOT / f"writing/fig/fig_t0_sensitivity_3series.{ext}"
    fig.savefig(p, dpi=150)
    print(f"saved: {p}")
print(f"saved: {out_csv}")

# -*- coding: utf-8 -*-
"""A-6: per-developer AUC ランキングに基づく IRL/RF 代表事例の可視化.

per-dev AUC を pair_trajectories.csv から自前で算出し、IRL 優位 / RF 優位 の代表 5 人ずつ
(計 10 人) について、4 horizon 跨ぎの軌跡を 1 枚パネルにまとめる。

評価制約:
  - bot 除外後の人間 dev のみ対象 (memory: bot_exclusion_policy)
  - 各 dev で IRL/RF 双方 AUC が計算可能 (label の両クラスが揃う) なものに限定
  - 観測数が少なすぎる dev (n_obs < 12) は除外

モデル (RULES.md 準拠): cold IRL (irl_<h>), RF (rf_<h>), アンサンブル ((irl+rf)/2)

入力:
  data/pair_trajectories.csv, data/pair_archetype.csv
  outputs/analysis/bot_detection/bot_candidates_main32.csv

出力:
  results/per_dev_auc.csv (再利用可能な per-dev AUC テーブル)
  results/case_study_devs.csv (選ばれた 10 人のメタ情報)
  results/fig_case_study_irl_top.pdf (IRL 優位 5 人パネル)
  results/fig_case_study_rf_top.pdf (RF 優位 5 人パネル)
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
REPO_ROOT = ROOT.parents[1]
BOT_CSV = REPO_ROOT / "outputs/analysis/bot_detection/bot_candidates_main32.csv"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
HORIZON_LABELS = ["0-3か月", "3-6か月", "6-9か月", "9-12か月"]
MIN_OBS = 12   # dev の最低観測数 (4 horizon × 3 dir)
TOP_K = 5


def add_ens(pair: pd.DataFrame) -> pd.DataFrame:
    pair = pair.copy()
    for h in HORIZONS:
        pair[f"ens_{h}"] = (pair[f"irl_{h}"] + pair[f"rf_{h}"]) / 2
    return pair


def per_dev_auc(pair: pd.DataFrame) -> pd.DataFrame:
    """各 dev について 4 horizon × all dirs の観測を pool した AUC."""
    rows = []
    for dev, g in pair.groupby("developer"):
        ys, irl_p, rf_p, ens_p = [], [], [], []
        for h in HORIZONS:
            ys.append(g[f"label_{h}"].values)
            irl_p.append(g[f"irl_{h}"].values)
            rf_p.append(g[f"rf_{h}"].values)
            ens_p.append(g[f"ens_{h}"].values)
        y = np.concatenate(ys)
        if len(np.unique(y)) < 2 or len(y) < MIN_OBS:
            continue
        rows.append({
            "developer": dev,
            "n_dir": g["directory"].nunique(),
            "n_obs": len(y),
            "n_pos": int(y.sum()),
            "base_rate": float(y.mean()),
            "auc_irl": roc_auc_score(y, np.concatenate(irl_p)),
            "auc_rf": roc_auc_score(y, np.concatenate(rf_p)),
            "auc_ens": roc_auc_score(y, np.concatenate(ens_p)),
        })
    df = pd.DataFrame(rows)
    df["irl_minus_rf"] = df["auc_irl"] - df["auc_rf"]
    return df.sort_values("irl_minus_rf", ascending=False).reset_index(drop=True)


def fetch_dev_summary(pair: pd.DataFrame, dev: str) -> dict:
    """1 人の dev について 4 horizon × 全 dir の平均軌跡を返す."""
    g = pair[pair["developer"] == dev]
    out = {"developer": dev, "n_dir": g["directory"].nunique(),
           "n_obs": len(g) * len(HORIZONS)}
    for h in HORIZONS:
        out[f"label_{h}"] = float(g[f"label_{h}"].mean())
        out[f"irl_{h}"] = float(g[f"irl_{h}"].mean())
        out[f"rf_{h}"] = float(g[f"rf_{h}"].mean())
        out[f"ens_{h}"] = float(g[f"ens_{h}"].mean())
        out[f"n_pos_{h}"] = int(g[f"label_{h}"].sum())
    return out


def plot_panel(devs: list[dict], dev_aucs: pd.DataFrame,
               arch: pd.DataFrame | None, title: str,
               out_path: pathlib.Path) -> None:
    n = len(devs)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    x = np.arange(len(HORIZONS))
    for ax, d in zip(axes, devs):
        dev = d["developer"]
        irl = [d[f"irl_{h}"] for h in HORIZONS]
        rf = [d[f"rf_{h}"] for h in HORIZONS]
        ens = [d[f"ens_{h}"] for h in HORIZONS]
        lab = [d[f"label_{h}"] for h in HORIZONS]

        ax.plot(x, irl, marker="o", color="#1f77b4", lw=2.2, label="cold IRL")
        ax.plot(x, rf, marker="s", color="#d62728", lw=2.2, label="RF")
        ax.plot(x, ens, marker="^", color="#2ca02c", lw=1.8,
                linestyle="--", label="アンサンブル")
        # label rate を bar で背景に
        ax.bar(x, lab, color="#cccccc", alpha=0.45, width=0.55,
               edgecolor="gray", label="label 平均")

        row = dev_aucs[dev_aucs["developer"] == dev].iloc[0]
        # archetype (任意)
        arch_str = ""
        if arch is not None:
            sub = arch[arch["developer"] == dev]
            if not sub.empty:
                arch_str = f", arch={sub.iloc[0]['dominant']}"

        # dev 名は @ で省略
        local = dev.split("@")[0]
        ax.set_title(
            f"{local}\n"
            f"AUC IRL={row['auc_irl']:.2f} / RF={row['auc_rf']:.2f}"
            f" (Δ={row['irl_minus_rf']:+.2f}{arch_str})\n"
            f"n_dir={int(row['n_dir'])}, n_pos={int(row['n_pos'])}/"
            f"{int(row['n_obs'])}",
            fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(HORIZON_LABELS, fontsize=8, rotation=20)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("予測確率 / label 平均 (dev × 全 dir × horizon)")
    axes[-1].legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    print("loading data...")
    pair = pd.read_csv(DATA / "pair_trajectories.csv")
    pair = add_ens(pair)
    bot_df = pd.read_csv(BOT_CSV)
    bots = set(bot_df.loc[bot_df["is_bot"] == True, "email"].tolist())  # noqa: E712
    pair_nb = pair[~pair["developer"].isin(bots)].copy()
    print(f"pair n={len(pair):,} → bot 除外 n={len(pair_nb):,}")

    # archetype 任意ロード (1 ペア 1 archetype の dominant を dev 単位に集約)
    arch_path = DATA / "pair_archetype.csv"
    arch_dev = None
    if arch_path.exists():
        ar = pd.read_csv(arch_path)
        if "dominant" in ar.columns:
            arch_dev = (ar.groupby("developer")["dominant"]
                        .agg(lambda s: s.mode().iat[0]).reset_index())

    dev_aucs = per_dev_auc(pair_nb)
    OUT.mkdir(exist_ok=True)
    dev_aucs.to_csv(OUT / "per_dev_auc.csv", index=False)
    print(f"per_dev_auc: n_dev={len(dev_aucs):,}, "
          f"IRL>RF n={int((dev_aucs['irl_minus_rf'] > 0).sum())}, "
          f"RF>IRL n={int((dev_aucs['irl_minus_rf'] < 0).sum())}")

    # top 5 IRL 優位, top 5 RF 優位
    irl_top = dev_aucs.head(TOP_K)
    rf_top = dev_aucs.tail(TOP_K).iloc[::-1]
    print("\n[IRL 優位 top 5]")
    print(irl_top[["developer", "n_obs", "n_pos", "auc_irl", "auc_rf",
                   "irl_minus_rf"]].round(3).to_string(index=False))
    print("\n[RF 優位 top 5]")
    print(rf_top[["developer", "n_obs", "n_pos", "auc_irl", "auc_rf",
                  "irl_minus_rf"]].round(3).to_string(index=False))

    chosen = pd.concat([irl_top.assign(side="IRL"),
                        rf_top.assign(side="RF")])
    chosen.to_csv(OUT / "case_study_devs.csv", index=False)

    irl_devs = [fetch_dev_summary(pair_nb, d) for d in irl_top["developer"]]
    rf_devs = [fetch_dev_summary(pair_nb, d) for d in rf_top["developer"]]

    plot_panel(irl_devs, dev_aucs, arch_dev,
               "IRL 優位 top 5: cold IRL / RF / アンサンブル の 4 horizon 軌跡",
               OUT / "fig_case_study_irl_top.pdf")
    plot_panel(rf_devs, dev_aucs, arch_dev,
               "RF 優位 top 5: cold IRL / RF / アンサンブル の 4 horizon 軌跡",
               OUT / "fig_case_study_rf_top.pdf")


if __name__ == "__main__":
    main()

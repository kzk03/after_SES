# -*- coding: utf-8 -*-
"""t0 掃引のアーキタイプ横断比較（追加分析のアーキタイプ最終集計）.

t0_compare.py が headline(AUC) の横断版なら、これは archetype の横断版。
各 t0 の archetype 出力 (run_t0_archetype.sh が産む outputs/t0_sweep/<tag>/archetype/)
を読み、基準 t0=2023-01 のアーキタイプにプロファイル最近傍で対応付け（Hungarian）
してから、t0 を動かしても

  「どの人物像/応募状況のアーキタイプで IRL が勝つ / RF が勝つ」地形

が安定かを表と図にする。IRL 出力が未着で archetype が未生成の t0 は自動 skip。

対応付けの考え方:
    各 t0 で archetype の index (A1..Ak) は seed 依存で順不同。基準 t0 の
    archetype プロファイル (k × feat) を参照に、他 t0 の archetype を
    Euclidean 距離最小のマッチングで並べ替え、同じ列が同じ人物像を指すようにする。

出力:
    outputs/t0_sweep/t0_archetype_<level>_k<k>_comparison.csv
    writing/fig/fig_t0_archetype_<level>_k<k>.pdf
"""
from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# [align系列] ALIGN=1 で outputs/t0_sweep_align_fs24/ を参照、出力は _align サフィックス。
ALIGN = os.environ.get("ALIGN", "0") == "1"
SWEEP = os.environ.get("T0_SWEEP_ROOT", "outputs/t0_sweep_align_fs24" if ALIGN else "outputs/t0_sweep")
SUFFIX = "_align" if ALIGN else ""
# t0_compare.py と同じ掃引点。archetype dir は tag から導出。
T0_CONFIG = [
    ("2022-01-01", "20220101", "year"),
    ("2022-10-01", "20221001", "3mo"),
    ("2023-01-01", "20230101", "base"),
    ("2023-04-01", "20230401", "3mo"),
    ("2024-01-01", "20240101", "year"),
]


def arch_dir(tag: str, level: str) -> pathlib.Path:
    return pathlib.Path(f"{SWEEP}/{tag}/archetype/{level}")


def pooled_auc_path(d: pathlib.Path, k: int) -> pathlib.Path:
    """dev は k{k}_pooled_auc_bootstrap.csv、pair は k{k}_pooled_auc.csv。"""
    for name in (f"k{k}_pooled_auc_bootstrap.csv", f"k{k}_pooled_auc.csv"):
        if (d / name).exists():
            return d / name
    return d / f"k{k}_pooled_auc_bootstrap.csv"


def load_t0(tag: str, level: str, k: int):
    d = arch_dir(tag, level)
    prof_f = d / f"k{k}_archetype_profile.csv"
    auc_f = pooled_auc_path(d, k)
    if not prof_f.exists() or not auc_f.exists():
        return None
    prof = pd.read_csv(prof_f, index_col=0)        # rows A1..Ak, cols=features
    auc = pd.read_csv(auc_f)                        # archetype, model, n_samples, auc, ...
    return prof, auc


def match_to_base(base_prof: pd.DataFrame, prof: pd.DataFrame) -> dict:
    """prof の各 archetype を base_prof に最近傍対応付け。{prof_arch -> base_arch}。"""
    B = base_prof.to_numpy()           # (k, f)
    P = prof.to_numpy()                # (k, f)
    # cost[i, j] = ||P_i - B_j||
    cost = np.linalg.norm(P[:, None, :] - B[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return {prof.index[i]: base_prof.index[j] for i, j in zip(row_ind, col_ind)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--level", choices=["dev", "pair"], default="dev")
    p.add_argument("--k", type=int, default=4)
    args = p.parse_args()
    level, k = args.level, args.k

    base = load_t0("20230101", level, k)
    if base is None:
        print(f"[error] 基準 t0=2023-01 の archetype 出力 (level={level}, k={k}) が無い。"
              f"先に `bash scripts/variant/run_t0_archetype.sh 2023-01-01` を実行してください。")
        return
    base_prof, _ = base

    rows = []
    for t0, tag, series in T0_CONFIG:
        loaded = load_t0(tag, level, k)
        if loaded is None:
            print(f"[skip] t0={t0}: archetype 未生成 ({arch_dir(tag, level)})")
            continue
        prof, auc = loaded
        mapping = match_to_base(base_prof, prof) if tag != "20230101" else {a: a for a in base_prof.index}

        # archetype × model の AUC を wide 化
        wide = auc.pivot_table(index="archetype", columns="model", values="auc")
        npiv = auc.pivot_table(index="archetype", columns="model", values="n_samples")
        for arch in prof.index:
            base_arch = mapping[arch]
            irl = wide.loc[arch].get("cold_irl", np.nan)
            rf = wide.loc[arch].get("rf_dir", np.nan)
            ens = wide.loc[arch].get("ens", np.nan)
            n = npiv.loc[arch].get("cold_irl", np.nan)
            rows.append({
                "t0": t0, "series": series, "base_archetype": base_arch,
                "local_archetype": arch, "n_samples": int(n) if pd.notna(n) else np.nan,
                "IRL": round(irl, 3) if pd.notna(irl) else np.nan,
                "RF": round(rf, 3) if pd.notna(rf) else np.nan,
                "Ens": round(ens, 3) if pd.notna(ens) else np.nan,
                "IRL_minus_RF": round(irl - rf, 3) if pd.notna(irl) and pd.notna(rf) else np.nan,
            })

    if not rows:
        print("集計対象なし（archetype 出力がまだ無い）。run_t0_archetype.sh の完了後に再実行。")
        return

    out = pd.DataFrame(rows).sort_values(["base_archetype", "t0"]).reset_index(drop=True)
    out_dir = pathlib.Path(SWEEP)
    csv_p = out_dir / f"t0_archetype_{level}_k{k}_comparison{SUFFIX}.csv"
    out.to_csv(csv_p, index=False)

    print(f"\n=== t0 掃引 × archetype ({level}-level, k={k}) : pooled AUC (B定義) ===")
    pd.set_option("display.width", 180)
    print(out.to_string(index=False))
    print(f"\nsaved: {csv_p}")

    # 図: base_archetype ごとに IRL−RF gap vs t0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import japanize_matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
        archs = sorted(out["base_archetype"].unique())
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for a in archs:
            sub = out[out["base_archetype"] == a].copy()
            sub["x"] = pd.to_datetime(sub["t0"])
            sub = sub.sort_values("x")
            ax.plot(sub["x"], sub["IRL_minus_RF"], marker="o", label=a)
        ax.axhline(0, ls="-", c="gray", lw=0.8)
        ax.axvline(pd.Timestamp("2023-01-01"), ls="--", c="gray", alpha=.6, label="t0 (paper)")
        ax.set_xlabel("t0 (snapshot time)")
        ax.set_ylabel("pooled AUC(IRL) − AUC(RF)\n + → IRL strong / − → RF strong")
        ax.set_title(f"Archetype別 IRL vs RF の t0 安定性 ({level}-level, k={k})")
        ax.legend(title="archetype (base対応)", fontsize=8)
        ax.grid(alpha=.3)
        fig.autofmt_xdate(); fig.tight_layout()
        figp = pathlib.Path(f"writing/fig/fig_t0_archetype_{level}_k{k}{SUFFIX}.pdf")
        figp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figp)
        print(f"saved: {figp}")
    except Exception as e:
        print(f"[warn] 図の生成をスキップ: {e}")


if __name__ == "__main__":
    main()

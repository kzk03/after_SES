# -*- coding: utf-8 -*-
"""t0 掃引の横断比較表を作る（追加分析の最終集計）.

各 t0 について IRL(diagonal)/RF の cold 予測を align し、B定義(承諾)ラベルで
pooled / horizon 別 AUC を再計算（[[blabel-recompute]] と同じ手順5）。
基準 2023-01 は既存 outputs を、その他は outputs/t0_sweep/<tag>/ を参照。
IRL 出力が未着の t0 は自動スキップ（GPU学習の途中でも回せる）。

出力:
  outputs/t0_sweep/t0_comparison_pooled.csv  … pooled の t0×model 表
  outputs/t0_sweep/t0_comparison_full.csv     … horizon 別も含む全行
  writing/fig/fig_t0_sensitivity.pdf           … pooled AUC vs t0 の折れ線
"""
from __future__ import annotations

import os
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from verify_blabel_headline import compute_headline, load_df_with_dirs  # noqa: E402

# [align系列] ALIGN=1 で outputs/t0_sweep_align/ を参照し、base(2023-01) も align 再学習物
# を使う（dynstartB は旧設計のため流用不可）。出力ファイル名は _align サフィックス。
# [inter系列] INTER=1 で outputs/t0_sweep_inter/（buffer消去のみ・tail統一なし）を参照。
# base も inter 再学習物を使う。出力ファイル名は _inter サフィックス。
ALIGN = os.environ.get("ALIGN", "0") == "1"
INTER = os.environ.get("INTER", "0") == "1"
if ALIGN and INTER:
    raise SystemExit("ALIGN=1 と INTER=1 は同時指定不可")
# 既定は各系列の従来 step 数（align=24step, inter=45step）。fs36/fs24 等の別 step
# 系列は T0_SWEEP_ROOT=outputs/t0_sweep_{align,inter}_fs<steps> で指定する。
_default_sweep = ("outputs/t0_sweep_align_fs24" if ALIGN
                  else "outputs/t0_sweep_inter_fs45" if INTER
                  else "outputs/t0_sweep")
SWEEP = os.environ.get("T0_SWEEP_ROOT", _default_sweep)
SUFFIX = "_align" if ALIGN else ("_inter" if INTER else "")

# t0 → (irl_base, rf_base, series ラベル)
if ALIGN or INTER:
    # 全点を同一系列で再学習（base 2023-01 も dynstartB 流用不可）
    _base_irl, _base_rf = f"{SWEEP}/20230101/irl/lstm_baseline", f"{SWEEP}/20230101/rf"
else:
    # 基準 2023-01 は既存 dynstartB を流用
    _base_irl, _base_rf = "outputs/dynstartB_ep150/lstm_baseline", "outputs/dynstart_rf_1snap_s777"
T0_CONFIG = [
    ("2022-01-01", f"{SWEEP}/20220101/irl/lstm_baseline", f"{SWEEP}/20220101/rf", "year"),
    ("2022-10-01", f"{SWEEP}/20221001/irl/lstm_baseline", f"{SWEEP}/20221001/rf", "3mo"),
    ("2023-01-01", _base_irl, _base_rf, "base"),
    ("2023-04-01", f"{SWEEP}/20230401/irl/lstm_baseline", f"{SWEEP}/20230401/rf", "3mo"),
    ("2024-01-01", f"{SWEEP}/20240101/irl/lstm_baseline", f"{SWEEP}/20240101/rf", "year"),
]


def irl_ready(irl_base: str) -> bool:
    return all(
        pathlib.Path(f"{irl_base}/train_{h}m/eval_{h}m/pair_predictions.csv").exists()
        for h in ["0-3", "3-6", "6-9", "9-12"]
    )


def main() -> None:
    df = load_df_with_dirs()
    full_rows, pooled_rows = [], []
    for t0, irl_base, rf_base, series in T0_CONFIG:
        if not irl_ready(irl_base):
            print(f"[skip] t0={t0}: IRL 出力が未着 ({irl_base})")
            continue
        print(f"[calc] t0={t0} ({series}) ...")
        tab = compute_headline(df, irl_base, rf_base, t0)
        tab.insert(0, "t0", t0)
        tab.insert(1, "series", series)
        full_rows.append(tab)
        prow = tab[tab["horizon"] == "pooled"].iloc[0]
        pooled_rows.append({
            "t0": t0, "series": series, "n": int(prow["n"]),
            "posrate": round(prow["posrate"], 3),
            "IRL": round(prow["IRL_roc"], 3), "RF": round(prow["RF_roc"], 3),
            "Ens": round(prow["Ens_roc"], 3),
            "Ens_minus_best": round(prow["Ens_roc"] - max(prow["IRL_roc"], prow["RF_roc"]), 3),
        })

    if not pooled_rows:
        print("\n集計対象なし（IRL 出力がまだ無い）。GPU学習の完了後に再実行してください。")
        return

    pooled = pd.DataFrame(pooled_rows).sort_values("t0").reset_index(drop=True)
    full = pd.concat(full_rows, ignore_index=True)
    out_dir = pathlib.Path(SWEEP); out_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(out_dir / f"t0_comparison_pooled{SUFFIX}.csv", index=False)
    full.to_csv(out_dir / f"t0_comparison_full{SUFFIX}.csv", index=False)

    print(f"\n=== t0 掃引{'(align)' if ALIGN else ''}: pooled AUC-ROC (B定義) ===")
    print(pooled.to_string(index=False))
    print(f"\nsaved: {out_dir/('t0_comparison_pooled'+SUFFIX+'.csv')} / t0_comparison_full{SUFFIX}.csv")

    # 図: pooled AUC vs t0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = pd.to_datetime(pooled["t0"])
        fig, ax = plt.subplots(figsize=(6, 4))
        for col, mk in [("IRL", "o"), ("RF", "s"), ("Ens", "^")]:
            ax.plot(x, pooled[col], marker=mk, label=col)
        ax.axvline(pd.Timestamp("2023-01-01"), ls="--", c="gray", alpha=.6, label="t0 (paper)")
        ax.set_xlabel("t0 (snapshot time)"); ax.set_ylabel("pooled AUC-ROC")
        ax.set_ylim(0.7, 0.86); ax.legend(); ax.grid(alpha=.3)
        fig.autofmt_xdate(); fig.tight_layout()
        figp = pathlib.Path(f"writing/fig/fig_t0_sensitivity{SUFFIX}.pdf")
        figp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figp); print(f"saved: {figp}")
    except Exception as e:  # 図は無くても表は出す
        print(f"[warn] 図の生成をスキップ: {e}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""SES査読B-3: 退化版(final_step_only)ベースラインのペアレベル分析.

実験ノート: Obsidian「2026-07-14_SES査読B3_退化版ベースライン比較」。
headline 集計 (verify_blabel_headline.py) の上に、以下を追加で行う:

1) 全モデルの予測を (developer, directory, horizon) で横結合した wide 表を作る
2) ペア・ブートストラップで pooled ΔAUC の 95%CI を出す (IRL本体 vs 各退化版 / RF)
3) horizon 内 rank 正規化で pooled を再計算し、pooled 低下が horizon 間
   スコア尺度の不揃いによるものかを検証する (Transformer で +0.033 回復)
4) horizon 別のスコア分布統計と、IRL本体との Spearman 相関
   (固定seed学習なので、フラグ失効なら本体と相関~1になるはず=provenance検証)

リポジトリルートから実行する:
    uv run python scripts/analyze/study/ses_b3_ablation_analysis.py

注意: ブートストラップCIはペアの標本分散のみで、seed分散(各モデル単一seed)は
含まれない。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from verify_blabel_headline import FS, HZ, load_df_with_dirs, recompute_blabel  # noqa: E402

MODELS = {
    "irl_main": "outputs/dynstartB_ep150/lstm_baseline",
    "lstm_fo": "outputs/ses_review/b3_ablation/lstm_final_only",
    "gru_fo": "outputs/ses_review/b3_ablation/gru_final_only",
    "tr_fo": "outputs/ses_review/b3_ablation/transformer_final_only",
}


def build_wide_table(df_dirs: pd.DataFrame, rf_base: str, t0: pd.Timestamp) -> pd.DataFrame:
    """label + 全モデル確率 + RF をペアレベルで横結合した表を返す。"""
    frames = []
    for h in HZ:
        dir_devs = recompute_blabel(df_dirs, t0, FS[h], 3, assign=False)
        rf = pd.read_csv(f"{rf_base}/fs_{h.replace('-', '_to_')}m/pair_predictions.csv")
        m = rf[["developer", "directory", "rf_dir_prob"]].copy()
        for name, base in MODELS.items():
            irl = pd.read_csv(f"{base}/train_{h}m/eval_{h}m/pair_predictions.csv")
            m = m.merge(
                irl[["developer", "directory", "irl_dir_prob"]].rename(
                    columns={"irl_dir_prob": f"p_{name}"}),
                on=["developer", "directory"], how="inner")
        m["horizon"] = h
        m["label"] = [
            1 if dev in dir_devs.get(d, set()) else 0
            for dev, d in zip(m["developer"], m["directory"])
        ]
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/combined_raw_main32.csv")
    p.add_argument("--raw-json-list", default="data/combined_raw_main32.raw_json_list.txt")
    p.add_argument("--rf-base", default="outputs/dynstart_rf_1snap_s777")
    p.add_argument("--prediction-time", default="2023-01-01")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv",
                   default="outputs/ses_review/b3_ablation/pooled_pairs_blabel_t0_20230101.csv")
    args = p.parse_args()

    t0 = pd.Timestamp(args.prediction_time)
    df = load_df_with_dirs(args.data, args.raw_json_list)
    W = build_wide_table(df, args.rf_base, t0)
    W.to_csv(args.out_csv, index=False)
    print(f"wide表 {W.shape} saved: {args.out_csv}")

    cols = {"irl_main": "p_irl_main", "lstm_fo": "p_lstm_fo",
            "gru_fo": "p_gru_fo", "tr_fo": "p_tr_fo", "rf": "rf_dir_prob"}
    y = W["label"].to_numpy()

    print("\n=== pooled AUC: 生スコア vs horizon内rank正規化 ===")
    for name, c in cols.items():
        raw = roc_auc_score(y, W[c])
        normed = roc_auc_score(y, W.groupby("horizon")[c].rank(pct=True))
        print(f"  {name:9s}: raw={raw:.3f}  rank-normalized={normed:.3f}  (diff {normed - raw:+.3f})")

    print("\n=== horizon別スコア分布 mean±std [q05, q95] ===")
    for name, c in cols.items():
        parts = []
        for h in HZ:
            s = W.loc[W["horizon"] == h, c]
            parts.append(
                f"{h}: {s.mean():.3f}±{s.std():.3f} [{s.quantile(0.05):.3f},{s.quantile(0.95):.3f}]")
        print(f"  {name:9s}: " + " | ".join(parts))

    print("\n=== Spearman相関 (horizon別, vs irl_main / 退化版同士) ===")
    for h in HZ:
        m = W[W["horizon"] == h]
        r = {k: m["p_irl_main"].corr(m[c], method="spearman")
             for k, c in cols.items() if k != "irl_main"}
        r["lstm_fo-gru_fo"] = m["p_lstm_fo"].corr(m["p_gru_fo"], method="spearman")
        print(f"  {h}: " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()))

    rng = np.random.default_rng(args.seed)
    n = len(W)
    P = {k: W[c].to_numpy() for k, c in cols.items()}
    pairs = [("irl_main", "lstm_fo"), ("irl_main", "gru_fo"), ("irl_main", "tr_fo"),
             ("irl_main", "rf"), ("lstm_fo", "rf"), ("lstm_fo", "gru_fo"), ("lstm_fo", "tr_fo")]
    deltas = {pr: np.empty(args.n_boot) for pr in pairs}
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        aucs = {k: roc_auc_score(yb, v[idx]) for k, v in P.items()}
        for a, bb in pairs:
            deltas[(a, bb)][b] = aucs[a] - aucs[bb]
    print(f"\n=== ペア・ブートストラップ ΔAUC (pooled, B={args.n_boot}, seed={args.seed}) ===")
    for (a, bb), d in deltas.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        point = roc_auc_score(y, P[a]) - roc_auc_score(y, P[bb])
        print(f"  {a} − {bb}: Δ={point:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  P(Δ≤0)={(d <= 0).mean():.4f}")

    print("\n=== horizon別 AUC ===")
    rows = []
    for h in HZ:
        mh = W[W["horizon"] == h]
        yh = mh["label"].to_numpy()
        rows.append({"horizon": h, **{k: roc_auc_score(yh, mh[c]) for k, c in cols.items()}})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

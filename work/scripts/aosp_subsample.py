"""AOSP の AUC 優位がデータ量によるものか文化によるものかを切り分ける。

OpenStack と同じ学習規模 (15,285) にサブサンプリングして測る。
時間分割は保ったまま、直近 n 件だけを学習に使う。
"""
import re, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from review_predictor.RL.env.acceptance_model import AcceptanceModel, binary_metrics

d = pickle.load(open("work/aosp_ep_7d.pkl", "rb"))
test_start = pd.Timestamp(d["episodes"][-d["meta"]["n_test"]]["start"])
t = pd.read_parquet("work/aosp_train.parquet").sort_values("added_at").reset_index(drop=True)
F = [c for c in t.columns if re.fullmatch(r"f\d{2}", c)]
tr_all, te = t[t.added_at < test_start], t[t.added_at >= test_start]
print(f"AOSP 全体: 学習 {len(tr_all):,} / 評価 {len(te):,}", flush=True)
print(f"\n{'学習n':>9}{'AUC':>9}{'PR-AUC':>9}{'Brier':>9}{'ECE':>9}", flush=True)
for n in (15285, 30000, 60000, len(tr_all)):
    n = min(n, len(tr_all))
    tr = tr_all.iloc[-n:]
    n_cal = int(len(tr) * 0.15)
    fit, cal = tr.iloc[:-n_cal], tr.iloc[-n_cal:]
    m = AcceptanceModel(obs_dim=26)
    m.fit(fit[F].to_numpy(np.float32), fit.contributed.to_numpy(),
          cal[F].to_numpy(np.float32), cal.contributed.to_numpy())
    p = m.predict_proba(te[F].to_numpy(np.float32))
    mm = binary_metrics(te.contributed.to_numpy(), p)
    print(f"{n:>9,}{mm['auc_roc']:>9.4f}{mm['auc_pr']:>9.4f}"
          f"{mm['brier']:>9.4f}{mm['ece']:>9.4f}", flush=True)
print("\n参考: OpenStack は学習 15,285 で AUC 0.7133")

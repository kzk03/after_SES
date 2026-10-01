"""貢献の判定期間 H の感度分析。

特徴量は H に依存しない (割り当て時点の履歴から計算) ので、
ラベルだけ付け替えて貢献モデルを学習し直せば PPO を回さずに評価できる。
"""
import re, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from review_predictor.RL.env.acceptance_model import AcceptanceModel, binary_metrics

ev = pd.read_parquet("work/assign_events.parquet")
t = pd.read_parquet("work/contrib_train.parquet")
F = [c for c in t.columns if re.fullmatch(r"f\d{2}", c)]
d = pickle.load(open("work/assign_ep.pkl", "rb"))
test_start = pd.Timestamp(d["episodes"][-d["meta"]["n_test"]]["start"])

key = {(r.reviewer, pd.Timestamp(r.added_at)): r.lat_days for r in ev.itertuples()}
lat = np.array([key.get((r.reviewer, pd.Timestamp(r.added_at)), np.nan) for r in t.itertuples()])
print(f"学習データ {len(t):,} 行  遅延が引けた行 {np.isfinite(lat).sum():,} "
      f"(残りは無応答 = 貢献なし)\n")

base = (lat <= 14)
print(f"{'H':>4}{'貢献率':>9}{'14日版からの変化':>16}{'学習n':>8}{'評価n':>8}"
      f"{'AUC':>8}{'PR-AUC':>8}{'Brier':>8}{'ECE':>8}")
for H in (3, 7, 14, 30, 60):
    y = (lat <= H)
    flip = np.mean(y != base)
    tt = t.copy(); tt["y"] = y.astype(int)
    tr, te = tt[tt.added_at < test_start], tt[tt.added_at >= test_start]
    n_cal = int(len(tr) * 0.15); fit, cal = tr.iloc[:-n_cal], tr.iloc[-n_cal:]
    m = AcceptanceModel(obs_dim=26)
    m.fit(fit[F].to_numpy(np.float32), fit.y.to_numpy(),
          cal[F].to_numpy(np.float32), cal.y.to_numpy())
    p = m.predict_proba(te[F].to_numpy(np.float32))
    mm = binary_metrics(te.y.to_numpy(), p)
    print(f"{H:>4}{y.mean():>9.4f}{flip:>16.4f}{len(tr):>8,}{len(te):>8,}"
          f"{mm['auc_roc']:>8.4f}{mm['auc_pr']:>8.4f}{mm['brier']:>8.4f}{mm['ece']:>8.4f}")

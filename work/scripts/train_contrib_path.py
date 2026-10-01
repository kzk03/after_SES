"""貢献モデルを AcceptanceModel 形式で学習・保存する (テスト期間より前だけで学習)。"""
import re, sys, pickle, os, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from review_predictor.RL.env.acceptance_model import AcceptanceModel, binary_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/assign_ep.pkl")
ap.add_argument("--train", default="work/contrib_train_path.parquet")
ap.add_argument("--out-dir", default="outputs/rl/contrib_model_v3_path")
a = ap.parse_args()
d = pickle.load(open(a.episodes, "rb"))
n_test = d["meta"]["n_test"]; test_start = pd.Timestamp(d["episodes"][-n_test]["start"])
t = pd.read_parquet(a.train).sort_values("added_at").reset_index(drop=True)
F = [c for c in t.columns if re.fullmatch(r"f\d{2}", c)]
tr, te = t[t.added_at < test_start], t[t.added_at >= test_start]
print(f"テスト開始 {test_start:%Y-%m-%d}  学習 {len(tr):,} / 評価 {len(te):,}")
n_cal = int(len(tr)*0.15); fit, cal = tr.iloc[:-n_cal], tr.iloc[-n_cal:]
m = AcceptanceModel(obs_dim=30)
m.fit(fit[F].to_numpy(np.float32), fit.contributed.to_numpy(),
      cal[F].to_numpy(np.float32), cal.contributed.to_numpy())
p = m.predict_proba(te[F].to_numpy(np.float32)); mm = binary_metrics(te.contributed.to_numpy(), p)
print(f"[test] n={len(te):,} AUC={mm['auc_roc']:.4f} PR-AUC={mm['auc_pr']:.4f} "
      f"Brier={mm['brier']:.4f} ECE={mm['ece']:.4f} 実測={te.contributed.mean():.4f} 予測={p.mean():.4f}")
os.makedirs(a.out_dir, exist_ok=True)
m.save(f"{a.out_dir}/contrib_model.joblib")
# .joblib は再生成できるので git 管理外。指標は JSON で残す
# (既存の acceptance_model*/acceptance_metrics.json と揃える)
json.dump({
    "config": {"episodes": a.episodes, "train": a.train, "obs_dim": 30,
               "test_start": str(test_start), "cal_frac": 0.15},
    "train": {"n_fit": len(fit), "n_cal": len(cal),
              "contrib_rate_fit": float(fit.contributed.mean()),
              "contrib_rate_cal": float(cal.contributed.mean())},
    "test": {"n": len(te), "contrib_rate": float(te.contributed.mean()),
             "mean_pred": float(p.mean()), **{k: float(v) for k, v in mm.items()
                                              if isinstance(v, (int, float))}},
}, open(f"{a.out_dir}/contrib_metrics.json", "w"), ensure_ascii=False, indent=1)
print(f"保存 {a.out_dir}/contrib_model.joblib, contrib_metrics.json")

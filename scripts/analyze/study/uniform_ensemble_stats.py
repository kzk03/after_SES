"""Uniform Ensemble (= (p_irl + p_rf) / 2) の統計値を計算する。

論文 tab:overall / tab:devlevel / tab:arch_grid の Uniform Ensemble 行と
それに付随する bootstrap CI / Wilcoxon p-value / 効果量 r を出力する。

入力:
  - outputs/main32_mce_volat_ep150/<variant>/train_W/eval_W/pair_predictions.csv
    (W in 0-3m, 3-6m, 6-9m, 9-12m の対角 4 セル)
  - outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/dev_window_pooled_with_alpha.csv
    (developer x window -> dominant archetype)

出力 (output-dir 配下):
  - per_window.csv          : window 別 IRL/RF/Ens AUC + delta_best + 95% bootstrap CI
  - per_archetype.csv       : archetype x window 別 AUC + gain_vs_best
  - per_dev_aucs.csv        : 混合ラベル開発者の per-dev AUC
  - wilcoxon_results.csv    : Ensemble vs IRL / vs RF の paired Wilcoxon (Bonferroni 補正)
  - summary.txt             : 主要指標の人間可読サマリ
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load_diagonal(base: Path, variant: str) -> dict[str, pd.DataFrame]:
    """対角 4 セルの pair_predictions.csv を window -> DataFrame で返す。

    Ensemble 列 (irl_dir_prob_calibrated と rf_dir_prob の単純平均) を追加する。
    """
    frames: dict[str, pd.DataFrame] = {}
    for h in HORIZONS:
        p = base / variant / f"train_{h}" / f"eval_{h}" / "pair_predictions.csv"
        if not p.exists():
            print(f"  missing: {p}")
            continue
        d = pd.read_csv(p)
        d = d.dropna(subset=["irl_dir_prob_calibrated", "rf_dir_prob", "label"]).copy()
        d["ensemble"] = (d["irl_dir_prob_calibrated"] + d["rf_dir_prob"]) / 2.0
        frames[h] = d[
            ["developer", "directory", "label",
             "irl_dir_prob_calibrated", "rf_dir_prob", "ensemble"]
        ]
    return frames


def bootstrap_delta_ci(
    y: np.ndarray, p_ens: np.ndarray, p_irl: np.ndarray, p_rf: np.ndarray,
    n_boot: int = 1000, alpha: float = 0.05, seed: int = 777,
) -> dict[str, float]:
    """Ensemble AUC - max(IRL, RF) AUC の点推定と stratified bootstrap CI。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    auc_irl = roc_auc_score(y, p_irl)
    auc_rf = roc_auc_score(y, p_rf)
    auc_ens = roc_auc_score(y, p_ens)
    delta = auc_ens - max(auc_irl, auc_rf)

    deltas = []
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        try:
            ae = roc_auc_score(yb, p_ens[idx])
            ai = roc_auc_score(yb, p_irl[idx])
            ar = roc_auc_score(yb, p_rf[idx])
            deltas.append(ae - max(ai, ar))
        except Exception:
            continue

    deltas_arr = np.array(deltas)
    lo = float(np.quantile(deltas_arr, alpha / 2))
    hi = float(np.quantile(deltas_arr, 1 - alpha / 2))
    return {
        "n": int(n), "n_pos": int(y.sum()),
        "auc_irl": auc_irl, "auc_rf": auc_rf, "auc_ens": auc_ens,
        "delta_best": delta, "ci_low": lo, "ci_high": hi,
    }


def per_window_stats(frames: dict[str, pd.DataFrame], n_boot: int) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        if h not in frames:
            continue
        d = frames[h]
        y = d["label"].astype(int).to_numpy()
        stats = bootstrap_delta_ci(
            y,
            d["ensemble"].to_numpy(),
            d["irl_dir_prob_calibrated"].to_numpy(),
            d["rf_dir_prob"].to_numpy(),
            n_boot=n_boot,
        )
        stats["window"] = h
        rows.append(stats)
    return pd.DataFrame(rows)[
        ["window", "n", "n_pos",
         "auc_irl", "auc_rf", "auc_ens",
         "delta_best", "ci_low", "ci_high"]
    ]


def per_archetype_stats(
    frames: dict[str, pd.DataFrame], arche_csv: Path,
) -> pd.DataFrame:
    arche = pd.read_csv(arche_csv)
    assign = arche[["developer", "window", "dominant"]].drop_duplicates()
    rows = []
    for h in HORIZONS:
        if h not in frames:
            continue
        d = frames[h].merge(
            assign[assign["window"] == h][["developer", "dominant"]],
            on="developer", how="left",
        )
        for arch in ["A1", "A2", "A3", "A4"]:
            sub = d[d["dominant"] == arch]
            if len(sub) < 10 or sub["label"].nunique() < 2:
                continue
            y = sub["label"].astype(int).to_numpy()
            ai = roc_auc_score(y, sub["irl_dir_prob_calibrated"])
            ar = roc_auc_score(y, sub["rf_dir_prob"])
            ae = roc_auc_score(y, sub["ensemble"])
            rows.append({
                "archetype": arch, "window": h,
                "n": len(sub), "n_pos": int(y.sum()),
                "auc_irl": ai, "auc_rf": ar, "auc_ens": ae,
                "gain_vs_best": ae - max(ai, ar),
            })
    return pd.DataFrame(rows)


def per_dev_stats(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, float]]:
    """混合ラベル開発者の per-dev AUC と pooled dev AUC を計算。"""
    all_pairs = []
    for h, d in frames.items():
        e = d.copy()
        e["window"] = h
        all_pairs.append(e)
    big = pd.concat(all_pairs, ignore_index=True)

    dev_rows = []
    for dev, g in big.groupby("developer"):
        if g["label"].nunique() < 2:
            continue
        y = g["label"].astype(int).to_numpy()
        try:
            ai = roc_auc_score(y, g["irl_dir_prob_calibrated"])
            ar = roc_auc_score(y, g["rf_dir_prob"])
            ae = roc_auc_score(y, g["ensemble"])
        except ValueError:
            continue
        dev_rows.append({
            "developer": dev, "n": len(g),
            "auc_irl": ai, "auc_rf": ar, "auc_ens": ae,
        })
    dev_df = pd.DataFrame(dev_rows)

    mixed = set(dev_df["developer"])
    pool = big[big["developer"].isin(mixed)]
    yp = pool["label"].astype(int).to_numpy()
    pooled = {
        "auc_irl": roc_auc_score(yp, pool["irl_dir_prob_calibrated"]),
        "auc_rf": roc_auc_score(yp, pool["rf_dir_prob"]),
        "auc_ens": roc_auc_score(yp, pool["ensemble"]),
    }
    return dev_df, pooled


def wilcoxon_stats(dev_df: pd.DataFrame) -> pd.DataFrame:
    """per-dev AUC で Ensemble vs IRL / vs RF の paired Wilcoxon。Bonferroni 補正 (x2)。"""
    rows = []
    for ref in ["irl", "rf"]:
        x = dev_df["auc_ens"].to_numpy()
        y = dev_df[f"auc_{ref}"].to_numpy()
        diffs = x - y
        result = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        n_effective = int((diffs != 0).sum())
        r = float((np.sign(diffs).sum()) / max(n_effective, 1))
        rows.append({
            "comparison": f"Ensemble vs {ref.upper()}",
            "n": len(dev_df), "n_effective": n_effective,
            "W": float(result.statistic),
            "p_raw": float(result.pvalue),
            "p_bonferroni": float(min(result.pvalue * 2.0, 1.0)),
            "rank_biserial_r": r,
            "mean_diff_auc": float(diffs.mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=Path("outputs/main32_mce_volat_ep150"))
    ap.add_argument("--variant", default="lstm_baseline")
    ap.add_argument("--archetype-csv", type=Path,
                    default=Path("outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/dev_window_pooled_with_alpha.csv"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/analysis/uniform_ensemble_volat_ep150"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_diagonal(args.base, args.variant)
    if not frames:
        raise SystemExit(f"対角セルの pair_predictions.csv が見つからない: {args.base}/{args.variant}")

    per_win = per_window_stats(frames, args.n_boot)
    per_win.to_csv(args.output_dir / "per_window.csv", index=False)
    print("\n=== Per-window ===")
    print(per_win.to_string(index=False))

    per_arch = per_archetype_stats(frames, args.archetype_csv)
    per_arch.to_csv(args.output_dir / "per_archetype.csv", index=False)
    print("\n=== Per-archetype x window ===")
    print(per_arch.to_string(index=False))

    dev_df, pooled = per_dev_stats(frames)
    dev_df.to_csv(args.output_dir / "per_dev_aucs.csv", index=False)
    print(f"\n=== Per-dev (N={len(dev_df)}) ===")
    print(f"per-dev AUC mean : IRL={dev_df['auc_irl'].mean():.4f}  "
          f"RF={dev_df['auc_rf'].mean():.4f}  Ens={dev_df['auc_ens'].mean():.4f}")
    print(f"pooled dev AUC   : IRL={pooled['auc_irl']:.4f}  "
          f"RF={pooled['auc_rf']:.4f}  Ens={pooled['auc_ens']:.4f}")

    wilcox_df = wilcoxon_stats(dev_df)
    wilcox_df.to_csv(args.output_dir / "wilcoxon_results.csv", index=False)
    print("\n=== Wilcoxon paired test (per-dev AUC) ===")
    print(wilcox_df.to_string(index=False))

    with open(args.output_dir / "summary.txt", "w") as f:
        ens_mean = per_win["auc_ens"].mean()
        delta_mean = per_win["delta_best"].mean()
        f.write("Uniform Ensemble Stats Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"4-window mean AUC : {ens_mean:.4f}\n")
        f.write(f"4-window mean delta_best : {delta_mean:+.4f} pp\n")
        f.write(f"Per-dev N={len(dev_df)}\n")
        f.write(f"  per-dev AUC mean Ens={dev_df['auc_ens'].mean():.4f}\n")
        f.write(f"  pooled dev AUC   Ens={pooled['auc_ens']:.4f}\n\n")
        f.write("=== per_window ===\n")
        f.write(per_win.to_string(index=False) + "\n\n")
        f.write("=== per_archetype ===\n")
        f.write(per_arch.to_string(index=False) + "\n\n")
        f.write("=== wilcoxon ===\n")
        f.write(wilcox_df.to_string(index=False) + "\n")

    print(f"\n[done] outputs at {args.output_dir}")


if __name__ == "__main__":
    main()

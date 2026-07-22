# -*- coding: utf-8 -*-
"""dev原型の AUC を論文方式（③: ペア単位サンプルを開発者原型で切り出してプール）で集計.

背景（2026-07-06）: `t0_archetype.py` の dev-level pooled AUC は
「(dev, horizon) を max 集約した1点」をプールする方式（= revision_proposal の
② 開発者継続の型別版）で、論文の表7（開発者×予測区間の IRL−RF）とは集計単位が違う。
論文の表7 は **開発者の原型でペア(dev,dir)サンプルを切り出し、開発者に集約せず
プールする**（③方式, revision_proposal「補足: ①と②の違い」）。n が 341/554/1213/1639
（計3,747=289開発者の全ペア）であることが証拠。

このスクリプトは既存の archetype 出力（k4_dev_alpha_with_metrics.csv の dominant）と
pair_predictions から③方式の dev原型別 AUC を再計算する:
  - horizon 別 IRL/RF/Ens AUC と IRL−RF（論文表7の形式, pp）
  - pooled AUC + bootstrap 95%CI

出力:
  <root>/t0_archetype_dev_k4_pairpooled.csv
  writing/t0_archetype_tables_devpp{suffix}.md

使い方:
  uv run python scripts/analyze/study/t0_archetype_dev_pairpooled.py \
      [--root outputs/t0_sweep]
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

ROOT = pathlib.Path(__file__).resolve().parents[3]
T0S = ["2022-01-01", "2022-10-01", "2023-01-01", "2023-04-01", "2024-01-01"]
MODELS = {"IRL": "irl_dir_prob", "RF": "rf_dir_prob", "Ens": "ens"}


def bases_for(root: str, tag: str) -> tuple[str, str]:
    """(irl_base, rf_base)。論文系列の base 2023-01 のみ既存 outputs を流用。"""
    if root == "outputs/t0_sweep" and tag == "20230101":
        return "outputs/dynstartB_ep150/lstm_baseline", "outputs/dynstart_rf_1snap_s777"
    return f"{root}/{tag}/irl/lstm_baseline", f"{root}/{tag}/rf"


def pooled_ci(y: np.ndarray, p: np.ndarray, rng, n_boot: int = 1000) -> tuple[float, float, float]:
    auc = roc_auc_score(y, p)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    return auc, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/t0_sweep")
    args = ap.parse_args()
    root = args.root
    name = pathlib.Path(root).name.removeprefix("t0_sweep").removeprefix("_")
    suffix = f"_{name}" if name else ""
    rng = np.random.default_rng(777)

    df = load_df_with_dirs()
    rows = []
    md = [
        f"# dev原型 × IRL/RF/Ens（論文③方式: ペアを開発者原型で切り出しプール, {root}, k=4, B定義）",
        "",
        "> 自動生成: `t0_archetype_dev_pairpooled.py`。論文の表7と同じ集計単位",
        "> （n はペア×horizon。開発者に集約しない）。",
        f"> **pooled 列の `[lo, hi]` は bootstrap 95%信頼区間**であり、複数系列を",
        f"> まとめた範囲ではない（本ファイルは {root} 単体）。",
        "> ⚠ 原型は各 t0 の局所 index（A1..A4, t0間の対応は",
        "> `t0_archetype_dev_k4_comparison*.csv` の base_archetype / match_ok を参照）。",
        "",
    ]
    for t0s in T0S:
        tag = t0s.replace("-", "")
        irl_base, rf_base = bases_for(root, tag)
        alpha_f = ROOT / root / tag / "archetype/dev/k4_dev_alpha_with_metrics.csv"
        irl_ok = all(pathlib.Path(f"{irl_base}/train_{h}m/eval_{h}m/pair_predictions.csv").exists()
                     for h in HZ)
        if not alpha_f.exists() or not irl_ok:
            print(f"[skip] {t0s}: archetype または IRL 出力なし")
            continue
        dev_arch = pd.read_csv(alpha_f, index_col=0)["dominant"]
        t0 = pd.Timestamp(t0s)
        frames = []
        for h in HZ:
            irl = pd.read_csv(f"{irl_base}/train_{h}m/eval_{h}m/pair_predictions.csv")
            rf = pd.read_csv(f"{rf_base}/fs_{h.replace('-', '_to_')}m/pair_predictions.csv")
            m = irl[["developer", "directory", "irl_dir_prob"]].merge(
                rf[["developer", "directory", "rf_dir_prob"]], on=["developer", "directory"])
            dd = recompute_blabel(df, t0, FS[h], 3, assign=False)
            m["label"] = [1 if dv in dd.get(d, set()) else 0
                          for dv, d in zip(m.developer, m.directory)]
            m["ens"] = (m["irl_dir_prob"] + m["rf_dir_prob"]) / 2
            m["horizon"] = h
            m["arch"] = m["developer"].map(dev_arch)
            frames.append(m.dropna(subset=["arch"]))
        L = pd.concat(frames, ignore_index=True)
        print(f"[calc] {t0s}: pairs with dev-archetype = {len(L)} "
              f"(horizon毎 {len(L)//4})")

        md.append(f"## t0 = {t0s}\n")
        md.append("| 原型 | n/horizon | 指標 | 0-3m | 3-6m | 6-9m | 9-12m | pooled AUC [95%CI] |")
        md.append("|---|---|---|---|---|---|---|---|")
        for arch in sorted(L["arch"].unique()):
            sub = L[L["arch"] == arch]
            n_h = len(sub) // 4
            per = {}
            for mname, col in MODELS.items():
                aucs = []
                for h in HZ:
                    s = sub[sub["horizon"] == h]
                    aucs.append(roc_auc_score(s["label"], s[col])
                                if s["label"].nunique() > 1 else np.nan)
                auc_p, lo, hi = pooled_ci(sub["label"].to_numpy(),
                                          sub[col].to_numpy(), rng)
                per[mname] = (aucs, auc_p, lo, hi)
                rows.append({
                    "t0": t0s, "archetype": arch, "model": mname, "n_pairs_per_h": n_h,
                    **{f"auc_{h}": round(a, 4) for h, a in zip(HZ, aucs)},
                    "auc_pooled": round(auc_p, 4),
                    "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                })
            for mname in MODELS:
                aucs, auc_p, lo, hi = per[mname]
                cells = " | ".join(f"{a:.3f}" for a in aucs)
                md.append(f"| {arch} | {n_h} | {mname} | {cells} "
                          f"| {auc_p:.3f} [{lo:.3f}, {hi:.3f}] |")
            diff = [i - r for i, r in zip(per["IRL"][0], per["RF"][0])]
            cells = " | ".join(f"{d*100:+.1f}" for d in diff)
            md.append(f"| {arch} | {n_h} | **IRL−RF (pp)** | {cells} "
                      f"| {(per['IRL'][1]-per['RF'][1])*100:+.1f} |")
        md.append("")

    if not rows:
        print("集計対象なし")
        return
    out_csv = ROOT / root / "t0_archetype_dev_k4_pairpooled.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_md = ROOT / f"writing/t0_archetype_tables_devpp{suffix}.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"saved: {out_csv}\nsaved: {out_md}")


if __name__ == "__main__":
    main()

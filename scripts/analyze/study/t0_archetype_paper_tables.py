# -*- coding: utf-8 -*-
"""論文の表5/表7 と同一レイアウトの原型別テーブルを t0 掃引の各点で生成する.

レイアウト（論文・最終提出版）: 原型ごとに IRL / RF / Δ(=IRL−RF, pp) の3行、
列は horizon 4つ + ΣΔ（4horizonのΔ合計, 正なら太字）。
- pair-level = 論文表5: ペア原型（k4_pair_alpha_with_metrics.csv の dominant）で
  ペア(dev,dir)サンプルを切り出し、horizon 別にプール AUC。
- dev-level = 論文表7（③方式）: 開発者原型（k4_dev_alpha_with_metrics.csv の
  dominant）で**ペアサンプル**を切り出し、開発者に集約せずプール。

原型名は論文の解釈名（tab:pair-arch / tab:archetypes）を、基準 t0=2023-01 への
対応付け（t0_archetype_{level}_k4_comparison*.csv）経由で付ける。
対応付けが崩壊している点（match_ok=False）は ⚠ を付す。

出力: writing/t0_archetype_paper_style{suffix}.md
      <root>/t0_archetype_paper_style.csv（全数値 long 形式）

使い方:
  uv run python scripts/analyze/study/t0_archetype_paper_tables.py \
      [--root outputs/t0_sweep]
"""
from __future__ import annotations

import argparse
import glob
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

# 論文の解釈名（基準 t0=2023-01 の base index に対して）
PAPER_DEV_NAMES = {  # tab:archetypes（dev プロファイルは全系列同一）
    "A1": "A1 低活動型", "A2": "A2 短期・レビュー中心型",
    "A3": "A3 高承諾・高活動型", "A4": "A4 常時活動・低承諾型",
}
PAPER_PAIR_NAMES = {  # tab:pair-arch（論文系列 base の index 順）
    "A1": "P1 他プロジェクト経験者", "A2": "P2 関与低下型",
    "A3": "P3 高承諾型", "A4": "P4 低活動・低承諾型",
}
# align_fs36 は pair の base index が論文と入れ替わっている（Hungarian 検証済み:
# fs36 A1↔論文 A3, A3↔A1, A2/A4 は同じ。t0_archetype_discussion.md §2 追記）
PAIR_BASE_TO_PAPER = {
    "outputs/t0_sweep":            {"A1": "A1", "A2": "A2", "A3": "A3", "A4": "A4"},
    "outputs/t0_sweep_align_fs24": {"A1": "A1", "A2": "A2", "A3": "A3", "A4": "A4"},
    "outputs/t0_sweep_align_fs36": {"A1": "A3", "A2": "A2", "A3": "A1", "A4": "A4"},
}


def bases_for(root: str, tag: str) -> tuple[str, str]:
    if root == "outputs/t0_sweep" and tag == "20230101":
        return "outputs/dynstartB_ep150/lstm_baseline", "outputs/dynstart_rf_1snap_s777"
    return f"{root}/{tag}/irl/lstm_baseline", f"{root}/{tag}/rf"


def load_mapping(root: str, level: str) -> pd.DataFrame | None:
    hits = sorted(glob.glob(str(ROOT / root / f"t0_archetype_{level}_k4_comparison*.csv")))
    if not hits:
        return None
    df = pd.read_csv(hits[0])
    if "match_ok" not in df.columns:
        df["match_ok"] = True
    return df[["t0", "base_archetype", "local_archetype", "match_ok"]]


def paper_name(root: str, level: str, base_arch: str) -> str:
    if level == "dev":
        return PAPER_DEV_NAMES[base_arch]
    return PAPER_PAIR_NAMES[PAIR_BASE_TO_PAPER[root][base_arch]]


def build_long(df, root: str, tag: str, t0s: str) -> pd.DataFrame | None:
    irl_base, rf_base = bases_for(root, tag)
    if not all(pathlib.Path(f"{irl_base}/train_{h}m/eval_{h}m/pair_predictions.csv").exists()
               for h in HZ):
        return None
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
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def block(L: pd.DataFrame, arch_col: str, arch: str, name: str) -> tuple[list[str], list[dict]]:
    """1原型ぶんの IRL/RF/Δ 3行（md）と long 行を返す。"""
    sub = L[L[arch_col] == arch]
    n_h = len(sub) // 4
    aucs = {}
    for mname, col in MODELS.items():
        aucs[mname] = []
        for h in HZ:
            s = sub[sub["horizon"] == h]
            aucs[mname].append(roc_auc_score(s["label"], s[col])
                               if s["label"].nunique() > 1 else np.nan)
    delta = [(i - r) * 100 for i, r in zip(aucs["IRL"], aucs["RF"])]
    sig = sum(delta)
    sig_s = f"**{sig:+.1f}**" if sig > 0 else f"{sig:+.1f}"
    md = [
        f"| {name} ({n_h:,}) | IRL | " + " | ".join(f"{a:.3f}" for a in aucs["IRL"]) + " | |",
        "| | RF | " + " | ".join(f"{a:.3f}" for a in aucs["RF"]) + " | |",
        "| | Δ | " + " | ".join(f"{d:+.1f}" for d in delta) + f" | {sig_s} |",
    ]
    rows = []
    for mname in MODELS:
        for h, a in zip(HZ, aucs[mname]):
            rows.append({"archetype_local": arch, "name": name, "model": mname,
                         "horizon": h, "auc": round(a, 4), "n_per_h": n_h})
    return md, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/t0_sweep")
    args = ap.parse_args()
    root = args.root
    name_ = pathlib.Path(root).name.removeprefix("t0_sweep").removeprefix("_")
    suffix = f"_{name_}" if name_ else ""

    maps = {lv: load_mapping(root, lv) for lv in ["pair", "dev"]}
    df = load_df_with_dirs()
    md = [
        f"# 論文表5/表7 形式の原型別テーブル（{root}, k=4, B定義）",
        "",
        "> 自動生成: `t0_archetype_paper_tables.py`。Δ = (IRL−RF)×100 (pp)、",
        "> ΣΔ = 4horizon の Δ 合計（正なら太字）。horizon 表記は実装準拠で 0–3m 等",
        "> （論文の 1–3m 等と同じ窓）。原型名は論文の解釈名を基準 t0=2023-01 への",
        "> 対応付け経由で付与。⚠ = 対応付け崩壊（名前の同一性が保証されない）。",
        "> pair-level = 論文表5（ペア原型でペアを切り出し）、",
        "> dev-level = 論文表7（開発者原型で**ペア**を切り出し＝③方式）。",
        "",
    ]
    all_rows = []
    for t0s in T0S:
        tag = t0s.replace("-", "")
        L = build_long(df, root, tag, t0s)
        if L is None:
            print(f"[skip] {t0s}: IRL 出力なし")
            continue
        section = [f"## t0 = {t0s}\n"]
        for level, alpha_name, unit_cols in [
            ("pair", "k4_pair_alpha_with_metrics.csv", ["developer", "directory"]),
            ("dev", "k4_dev_alpha_with_metrics.csv", ["developer"]),
        ]:
            alpha_f = ROOT / root / tag / f"archetype/{level}/{alpha_name}"
            if not alpha_f.exists() or maps[level] is None:
                section.append(f"（{level}: archetype 出力なし）\n")
                continue
            al = pd.read_csv(alpha_f)
            if level == "dev":
                al = al.rename(columns={al.columns[0]: "developer"})
            key = al.set_index(unit_cols)["dominant"]
            X = L.copy()
            X["arch"] = X.set_index(unit_cols).index.map(key)
            X = X.dropna(subset=["arch"])
            mp = maps[level]
            mp_t = mp[mp["t0"] == t0s].set_index("local_archetype")
            title = "ペア単位（表5相当）" if level == "pair" else "開発者単位・③方式（表7相当）"
            section.append(f"### {title}\n")
            section.append("| 原型 (n/horizon) | 指標 | 0–3m | 3–6m | 6–9m | 9–12m | ΣΔ |")
            section.append("|---|---|---|---|---|---|---|")
            # 論文名の P1..P4 / A1..A4 順で出す
            order = sorted(mp_t.index, key=lambda a: paper_name(root, level, mp_t.loc[a, "base_archetype"]))
            for arch in order:
                base_a = mp_t.loc[arch, "base_archetype"]
                warn = "" if mp_t.loc[arch, "match_ok"] else " ⚠"
                nm = paper_name(root, level, base_a) + warn
                if tag != "20230101":
                    nm += f"〔局所{arch}〕"
                b_md, b_rows = block(X, "arch", arch, nm)
                section.extend(b_md)
                for r in b_rows:
                    r.update({"t0": t0s, "level": level})
                    all_rows.append(r)
            section.append("")
        md.extend(section)

    if not all_rows:
        print("集計対象なし")
        return
    out_csv = ROOT / root / "t0_archetype_paper_style.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    out_md = ROOT / f"writing/t0_archetype_paper_style{suffix}.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"saved: {out_csv}\nsaved: {out_md}")


if __name__ == "__main__":
    main()

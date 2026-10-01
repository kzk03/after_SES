"""D42: revert 以外の品質指標 — 後続の修正を誘発した PR(fix-inducing change)。

D35 は revert をアウトカムに使い、「AI 導入が revert を 30% 以上減らした可能性は棄却できる。
20% 以下は棄却できない」を得た。だが revert は粗い。**前進修正で直された欠陥は revert に現れない。**

より広い代理として、**後続の修正コミットに名指しされた PR** を使う。
コミットメッセージに fix 系の語と `#N` の参照が同時に現れるとき、
N はその修正が直した対象である可能性が高い。

⚠ 実装上の罠が2つある。
  1. **自己参照**: squash merge の慣習で「タイトル (#自分の番号)」が入る。必ず除く
  2. **時間順**: 参照先が後からマージされた PR なら因果が逆。マージ時刻で絞る
さらに精度を上げるため、**参照元と参照先でファイルが1つ以上重なる**ことを要求する。

出力: outputs_full/d42_fix_inducing.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d42_fix_inducing.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import OUT_DIR, RAW_DIR, fit_panel, load_repos, repo_dir_name

FIX = re.compile(r"\b(fix(e[sd])?|bug|regression|hotfix|broke[n]?)\b", re.I)
REF = re.compile(r"#(\d{2,7})")
FOLLOWUP_MONTHS = 3
MAX_GAP_DAYS = 180      # これより古い PR への参照は因果として弱いので切る


def month_to_idx(s: str) -> int:
    y, m = str(s).split("-")[:2]
    return int(y) * 12 + int(m)


def build() -> tuple[pd.DataFrame, dict]:
    rows = []
    stats = {"n_refs_raw": 0, "n_self": 0, "n_not_a_pr": 0,
             "n_wrong_order": 0, "n_too_old": 0, "n_no_file_overlap": 0, "n_linked": 0}

    for repo in load_repos():
        path = RAW_DIR / repo_dir_name(repo) / "prs_full.jsonl"
        if not path.exists():
            continue
        prs: dict[int, dict] = {}
        with path.open() as f:
            for line in f:
                try:
                    pr = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not pr.get("createdAt"):
                    continue
                files = {fl.get("path") for fl in (pr.get("files") or {}).get("nodes", [])
                         if fl.get("path")}
                msgs = [(c.get("commit") or {}).get("message") or ""
                        for c in (pr.get("commits") or {}).get("nodes", [])]
                prs[pr["number"]] = {
                    "number": pr["number"], "month": pr["createdAt"][:7],
                    "merged_at": pr.get("mergedAt"), "files": files, "msgs": msgs,
                    "merged": pr.get("mergedAt") is not None,
                }

        induced: set[int] = set()
        for num, p in prs.items():
            if not p["merged_at"]:
                continue
            t_self = datetime.fromisoformat(p["merged_at"].replace("Z", "+00:00"))
            for msg in p["msgs"]:
                if not FIX.search(msg):
                    continue
                for ref in REF.findall(msg):
                    stats["n_refs_raw"] += 1
                    n = int(ref)
                    if n == num:
                        stats["n_self"] += 1
                        continue
                    tgt = prs.get(n)
                    if tgt is None or not tgt["merged_at"]:
                        stats["n_not_a_pr"] += 1
                        continue
                    t_tgt = datetime.fromisoformat(tgt["merged_at"].replace("Z", "+00:00"))
                    if t_tgt >= t_self:
                        stats["n_wrong_order"] += 1
                        continue
                    if (t_self - t_tgt).days > MAX_GAP_DAYS:
                        stats["n_too_old"] += 1
                        continue
                    if not (p["files"] & tgt["files"]):
                        stats["n_no_file_overlap"] += 1
                        continue
                    induced.add(n)
                    stats["n_linked"] += 1

        for num, p in prs.items():
            rows.append({"repo": repo, "pr_number": num, "month": p["month"],
                         "merged": p["merged"], "fix_induced": float(num in induced)})
    return pd.DataFrame(rows), stats


def main() -> None:
    adoption = json.loads((OUT_DIR / "d4_repo_event_study.json").read_text())["adoption"]
    # ★対照は never-treated のみ。判定不能で除外したリポジトリを対照に混ぜない(2026-09-27 数値表の指摘)
    _d4 = json.loads((OUT_DIR / "d4_repo_event_study.json").read_text())
    KEEP_REPOS = set(adoption) | set(_d4["never_treated"])
    prs, stats = build()
    print("=== 参照の絞り込み ===")
    for k, v in stats.items():
        print(f"  {k:22s} {v:,}")
    print(f"  → 残った有効な参照: {stats['n_linked']:,}")

    d5 = pd.read_parquet(OUT_DIR / "d5_pr_level.parquet")[
        ["repo", "pr_number", "module_id_global", "n_ai_reviews_total",
         "n_human_reviews_total"]]
    df = prs.merge(d5, on=["repo", "pr_number"], how="inner")
    df = df[df["repo"].isin(KEEP_REPOS)]
    df["t"] = df["month"].map(month_to_idx)
    df = df[df["t"] <= df["t"].max() - FOLLOWUP_MONTHS]
    df["adopt_t"] = df["repo"].map(lambda r: month_to_idx(adoption[r]) if r in adoption else np.nan)
    df["post"] = (df["adopt_t"].notna() & (df["t"] >= df["adopt_t"])).astype(float)

    print(f"\n分析対象 PR {len(df):,} 件(打ち切り {FOLLOWUP_MONTHS} ヶ月)")
    print(f"  後続の修正に名指しされた PR: {int(df.fix_induced.sum()):,} 件 "
          f"({df.fix_induced.mean():.3%})")

    print("\n=== 記述: レビュー構成 × 修正誘発率(選択があるので因果ではない)===")
    df["got_ai"] = (df["n_ai_reviews_total"] > 0).astype(float)
    df["got_hu"] = (df["n_human_reviews_total"] > 0).astype(float)
    g = df.groupby(["got_hu", "got_ai"]).agg(
        n=("fix_induced", "size"), 修正誘発率=("fix_induced", "mean")).reset_index()
    g["型"] = np.where(g.got_hu.eq(0) & g.got_ai.eq(0), "レビュー無し",
              np.where(g.got_hu.eq(0) & g.got_ai.eq(1), "AIのみ",
              np.where(g.got_hu.eq(1) & g.got_ai.eq(0), "人間のみ", "両方")))
    print(g[["型", "n", "修正誘発率"]].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    mm = df.groupby(["repo", "module_id_global", "month"]).agg(
        n_prs=("fix_induced", "size"), fix_rate=("fix_induced", "mean"),
        n_fix=("fix_induced", "sum")).reset_index()
    mm["t"] = mm["month"].map(month_to_idx)
    mm["adopt_t"] = mm["repo"].map(lambda r: month_to_idx(adoption[r]) if r in adoption else np.nan)
    mm["post"] = (mm["adopt_t"].notna() & (mm["t"] >= mm["adopt_t"])).astype(float)
    mm["month_period"] = pd.PeriodIndex(mm["month"], freq="M").to_timestamp()

    out: dict = {"link_stats": stats, "n_prs": int(len(df)),
                 "fix_induced_rate": float(df["fix_induced"].mean()), "models": {}}

    print(f"\nモジュール×月 {len(mm):,} 行 / 修正誘発率の平均 {mm['fix_rate'].mean():.4%}")
    print("\n=== ★AI 導入は修正誘発率を下げるか ===")
    base = float(mm["fix_rate"].mean())
    r = fit_panel(mm, "fix_rate", ["post"])
    c, se = r["params"]["post"], r["se_repo"]["post"]
    star = " *" if r["p_repo"]["post"] < 0.05 else ""
    print(f"  fix_rate  {c:+.5f} (se={se:.5f}, p={r['p_repo']['post']:.4g}) n={r['n_obs']:,}{star}")
    mde = (1.96 + 1.64) * se
    print(f"  基準率 {base:.4%} / MDE {mde:.5f} (相対 {mde/base:.1%})")
    for d_rel in [0.10, 0.20, 0.30]:
        delta = d_rel * base
        p_one = 1 - norm.cdf((c + delta) / se)
        print(f"    「{d_rel:.0%}以上の減少がある」: p={p_one:.4g} -> "
              f"{'棄却できる' if p_one < 0.05 else '棄却できない'}")
        out.setdefault("equivalence", {})[f"{d_rel:.0%}"] = float(p_one)
    out["models"]["fix_rate"] = r
    out["base_rate"] = base

    (OUT_DIR / "d42_fix_inducing.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float))
    print(f"\n-> {OUT_DIR / 'd42_fix_inducing.json'}")


if __name__ == "__main__":
    main()

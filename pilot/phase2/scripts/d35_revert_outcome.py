"""D35: 増えた人間の労力で何が買えたのか — revert をアウトカムにする。

本研究はここまで「AI を入れても人間の作業は減らない」を示してきた。
D33/D34 はさらに、**疎なモジュールでは 1 PR あたりの人間のレビューが増える**
(+0.673, p=0.000)ことを示した。では**その増えた労力で何が買えたのか**。

レビューの目的は欠陥を止めることである。止め損ねた欠陥の観測可能な代理が revert である。

  「この PR は後で revert されたか」= レビューをすり抜けた欠陥

revert コミットのメッセージ "This reverts commit <sha>" から元コミットの sha を取り、
sha → PR の対応表で**元の PR に逆引きする**。これにより
「revert された PR」を PR 単位のアウトカムとして構成できる。

問い:
  1. AI 導入は revert 率を下げるか(リポジトリ単位の導入イベントで識別)
  2. 下げるなら、増えた労力には見返りがある
  3. 下げないなら、**増えた労力は何も買っていない**

⚠ 打ち切り: 直近の PR は revert される時間が足りない。
   観測終端から3ヶ月以上前に作成された PR に限定する。

⚠ 検出力: 逆引きできた「revert された PR」は 0.16% しかなく、これだけでは
   相対 50% 超の効果しか検出できない。そこで**主指標を revert PR の発生率**
   (その module-month の PR のうち revert PR が占める割合、~1.5%)に置く。
   逆引き版は頑健性確認として併記する。
   さらに null を主張するため、**同値性検定(片側 TOST)で
   「δ 以上の減少は無い」と言えるかを出す**(D9 と同じ考え方)。

出力: outputs_full/d35_revert_outcome.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d35_revert_outcome.py
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import OUT_DIR, RAW_DIR, load_repos, repo_dir_name, fit_panel

# "Revert \"...\"" で始まる、または本文に "This reverts commit <sha>" を含む
REVERT_HEAD = re.compile(r"^revert\b", re.I)
REVERTS_SHA = re.compile(r"this reverts commit ([0-9a-f]{7,40})", re.I)

FOLLOWUP_MONTHS = 3      # 打ち切り対策: 観測終端から何ヶ月前までを対象にするか


def month_to_idx(s: str) -> int:
    y, m = str(s).split("-")[:2]
    return int(y) * 12 + int(m)


def build_revert_links() -> tuple[pd.DataFrame, dict]:
    """sha -> PR の対応表を作り、revert された PR を特定する。"""
    sha_to_pr: dict[tuple[str, str], int] = {}
    revert_targets: list[tuple[str, str]] = []   # (repo, 元コミットの sha)
    pr_rows = []
    n_revert_pr = 0

    for repo in load_repos():
        path = RAW_DIR / repo_dir_name(repo) / "prs_full.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                try:
                    pr = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not pr.get("createdAt"):
                    continue
                num = pr["number"]
                commits = (pr.get("commits") or {}).get("nodes", [])
                msgs = []
                for c in commits:
                    cm = c.get("commit") or {}
                    oid = cm.get("oid")
                    if oid:
                        sha_to_pr[(repo, oid)] = num
                    msgs.append(cm.get("message") or "")
                rev_flags = [bool(REVERT_HEAD.search(m) or REVERTS_SHA.search(m)) for m in msgs]
                # ★「純粋な revert PR」= 全コミットが revert。
                #   revert を1つでも含むだけだと、PR 内の後始末(自分のコミットを戻す)を数えてしまう。
                #   2026-09-27 の査読(R2)で、旧定義の revert PR の 69% が他のコミットも含むと判明。
                is_revert = bool(rev_flags) and all(rev_flags)
                if is_revert:
                    n_revert_pr += 1
                for m in msgs:
                    for sha in REVERTS_SHA.findall(m):
                        revert_targets.append((repo, sha, num))
                pr_rows.append({
                    "repo": repo, "pr_number": num,
                    "month": pr["createdAt"][:7],
                    "merged": pr.get("mergedAt") is not None,
                    "is_revert_pr": is_revert,
                })
    prs = pd.DataFrame(pr_rows)

    # 短縮 sha にも対応するため、接頭辞でも引けるようにする
    by_repo: dict[str, dict[str, int]] = {}
    for (repo, oid), num in sha_to_pr.items():
        by_repo.setdefault(repo, {})[oid] = num

    reverted: set[tuple[str, int]] = set()
    n_hit = 0
    n_self = 0
    for repo, sha, src in revert_targets:
        table = by_repo.get(repo)
        if not table:
            continue
        num = table.get(sha)
        if num is None and len(sha) < 40:
            # 短縮 sha: 接頭辞一致を探す(線形探索だがリポジトリ単位なので許容)
            for oid, n in table.items():
                if oid.startswith(sha):
                    num = n
                    break
        if num is not None and num == src:
            n_self += 1          # 同じ PR の中で自分のコミットを戻している。欠陥ではない
            continue
        if num is not None:
            reverted.add((repo, num))
            n_hit += 1

    prs["was_reverted"] = [
        (r, n) in reverted for r, n in zip(prs["repo"], prs["pr_number"])
    ]
    stats = {
        "n_prs": int(len(prs)),
        "n_commits_indexed": len(sha_to_pr),
        "n_revert_prs": int(n_revert_pr),
        "n_revert_targets": len(revert_targets),
        "n_linked": int(n_hit),
        "link_rate": n_hit / max(len(revert_targets), 1),
        "n_reverted_prs": int(len(reverted)),
        "n_self_links_excluded": int(n_self),
    }
    return prs, stats


def main() -> None:
    adoption = json.loads((OUT_DIR / "d4_repo_event_study.json").read_text())["adoption"]
    # ★対照は never-treated のみ。判定不能で除外したリポジトリを対照に混ぜない(2026-09-27 数値表の指摘)
    _d4 = json.loads((OUT_DIR / "d4_repo_event_study.json").read_text())
    KEEP_REPOS = set(adoption) | set(_d4["never_treated"])

    prs, stats = build_revert_links()
    print("=== revert の逆引き ===")
    for k, v in stats.items():
        print(f"  {k:22s} {v:,}" if isinstance(v, int) else f"  {k:22s} {v:.3f}")

    # モジュールと AI レビューの有無を d5 から持ってくる
    d5 = pd.read_parquet(OUT_DIR / "d5_pr_level.parquet")[
        ["repo", "pr_number", "module_id_global", "n_ai_reviews_total",
         "n_human_reviews_total", "pr_lines", "pr_files"]]
    df = prs.merge(d5, on=["repo", "pr_number"], how="inner")
    df = df[df["repo"].isin(KEEP_REPOS)]
    df["t"] = df["month"].map(month_to_idx)
    end = df["t"].max()
    df = df[df["t"] <= end - FOLLOWUP_MONTHS]          # 打ち切り対策
    # revert PR 自体は「逆引き版」の母集団からは外すが、発生率の指標には要る
    df["adopt_t"] = df["repo"].map(lambda r: month_to_idx(adoption[r]) if r in adoption else np.nan)
    df["post"] = (df["adopt_t"].notna() & (df["t"] >= df["adopt_t"])).astype(float)
    df["was_reverted"] = df["was_reverted"].astype(float)

    print(f"\n分析対象 PR {len(df):,} 件(打ち切り {FOLLOWUP_MONTHS} ヶ月、revert PR 自体は除外)")
    print(f"  revert された PR: {int(df['was_reverted'].sum()):,} 件 ({df['was_reverted'].mean():.3%})")

    print("\n=== 記述: AI レビューの有無 × revert 率(選択があるので因果ではない)===")
    df["got_ai"] = (df["n_ai_reviews_total"] > 0).astype(float)
    df["got_hu"] = (df["n_human_reviews_total"] > 0).astype(float)
    g = df.groupby(["got_hu", "got_ai"]).agg(
        n=("was_reverted", "size"), revert率=("was_reverted", "mean"),
        マージ率=("merged", "mean"), 行数中央値=("pr_lines", "median")).reset_index()
    g["型"] = np.where(g.got_hu.eq(0) & g.got_ai.eq(0), "レビュー無し",
              np.where(g.got_hu.eq(0) & g.got_ai.eq(1), "AIのみ",
              np.where(g.got_hu.eq(1) & g.got_ai.eq(0), "人間のみ", "両方")))
    print(g[["型", "n", "revert率", "マージ率", "行数中央値"]].to_string(
        index=False, float_format=lambda x: f"{x:10.4f}"))

    # --- モジュール×月に集計してイベントスタディ ---
    df["is_revert_pr_f"] = df["is_revert_pr"].astype(float)
    mm = df.groupby(["repo", "module_id_global", "month"]).agg(
        n_prs=("was_reverted", "size"),
        revert_pr_rate=("is_revert_pr_f", "mean"),     # ★主指標(事象が多い)
        n_revert_prs=("is_revert_pr_f", "sum"),
        revert_rate=("was_reverted", "mean"),          # 逆引き版(頑健性)
        n_reverted=("was_reverted", "sum"),
        ai_share=("got_ai", "mean")).reset_index()
    mm["t"] = mm["month"].map(month_to_idx)
    mm["adopt_t"] = mm["repo"].map(lambda r: month_to_idx(adoption[r]) if r in adoption else np.nan)
    mm["post"] = (mm["adopt_t"].notna() & (mm["t"] >= mm["adopt_t"])).astype(float)
    mm["month_period"] = pd.PeriodIndex(mm["month"], freq="M").to_timestamp()
    mm["log_reverted"] = np.log1p(mm["n_reverted"])
    mm["log_revert_prs"] = np.log1p(mm["n_revert_prs"])

    print(f"\nモジュール×月 {len(mm):,} 行")
    print(f"  ★revert PR 発生率(主指標)の平均: {mm['revert_pr_rate'].mean():.4%}")
    print(f"  逆引き版 revert 率の平均:        {mm['revert_rate'].mean():.4%}")

    out: dict = {"link_stats": stats, "n_prs": int(len(df)),
                 "revert_rate_overall": float(df["was_reverted"].mean()), "models": {}}

    print("\n=== ★AI 導入は revert 率を下げるか(module + month 固定効果、repo クラスタ)===")
    base = {"revert_pr_rate": float(mm["revert_pr_rate"].mean()),
            "revert_rate": float(mm["revert_rate"].mean())}
    for y, lab in [("revert_pr_rate", "★revert PR 発生率"),
                   ("log_revert_prs", "  log revert PR 数"),
                   ("revert_rate", "  逆引き revert 率"),
                   ("log_reverted", "  log 逆引き件数")]:
        r = fit_panel(mm, y, ["post"])
        c, se = r["params"]["post"], r["se_repo"]["post"]
        star = " *" if r["p_repo"]["post"] < 0.05 else ""
        line = (f"  {lab:20s} {c:+.5f} (se={se:.5f}, p={r['p_repo']['post']:.4g}) "
                f"n={r['n_obs']:,}{star}")
        # 同値性: 「δ 以上の減少は無い」の片側検定と、検出可能な最小効果
        if y in base and base[y] > 0:
            mde = 1.96 * se + 1.64 * se          # 両側5%・検出力95%のおおよその MDE
            line += f"\n      基準率 {base[y]:.4%} / MDE {mde:.5f} (相対 {mde/base[y]:.1%})"
            for delta_rel in [0.10, 0.20, 0.30]:
                delta = delta_rel * base[y]
                z = (c + delta) / se            # H0: 効果 <= -delta
                from scipy.stats import norm
                p_one = 1 - norm.cdf(z)
                mark = "棄却できる" if p_one < 0.05 else "棄却できない"
                line += f"\n      「{delta_rel:.0%}以上の減少がある」: p={p_one:.4g} -> {mark}"
        print(line)
        out["models"][y] = r
    out["base_rates"] = base

    # --- D33 と同じ密度層別: 人間の労力が増えた層で revert は減ったか ---
    pre = mm[mm["adopt_t"].isna() | (mm["t"] < mm["adopt_t"])]
    # 密度は D33 と揃えて「導入前の人間レビュー件数の平均」を使う
    d5h = d5.merge(prs[["repo", "pr_number", "month"]], on=["repo", "pr_number"], how="inner")
    d5h["t"] = d5h["month"].map(month_to_idx)
    d5h["adopt_t"] = d5h["repo"].map(lambda r: month_to_idx(adoption[r]) if r in adoption else np.nan)
    # ★全モジュール共通の暦窓で密度を測る(d33 と同じ修正。非対称だと平均回帰が混入する)
    pre_h = d5h[d5h["month"] < "2025-01"]
    dens = pre_h.groupby("module_id_global")["n_human_reviews_total"].mean().rename("pre_density")
    mm = mm.merge(dens, on="module_id_global", how="left").dropna(subset=["pre_density"])
    mm["q"] = pd.qcut(mm["pre_density"], 4, labels=["Q1(疎)", "Q2", "Q3", "Q4(密)"],
                      duplicates="drop")
    print("\n=== 層別: 人間の労力が最も増えた Q1(疎)で revert は減ったか ===")
    out["by_density"] = {}
    for q in ["Q1(疎)", "Q2", "Q3", "Q4(密)"]:
        sub = mm[mm["q"] == q]
        if len(sub) < 500:
            continue
        r = fit_panel(sub, "revert_pr_rate", ["post"])
        star = " *" if r["p_repo"]["post"] < 0.05 else ""
        print(f"  {q:10s} {r['params']['post']:+.5f} "
              f"(p={r['p_repo']['post']:.4g}) n={r['n_obs']:,}{star}")
        out["by_density"][q] = r

    (OUT_DIR / "d35_revert_outcome.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float))
    print(f"\n-> {OUT_DIR / 'd35_revert_outcome.json'}")


if __name__ == "__main__":
    main()

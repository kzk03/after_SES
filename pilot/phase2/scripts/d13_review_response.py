"""D13: AI のレビューコメントは実際にコードを変えているのか。

D10/D11 で「AI レビューの 99.6% は COMMENTED で、判断をしていない」ことが分かった。
これに対する当然の反論は「判断しないのは仕様だ。だが助言としては価値があるはずだ」である。

そこで、レビューが実際にコード変更につながったかを測る。
GitHub の PR には commit の committedDate があるので、
「そのレビューが出た後に、この PR に新しい commit が入ったか」で応答を近似できる。

公平な比較のため **state を揃える**。人間も AI も COMMENTED を出すので、
「人間の COMMENTED」と「AI の COMMENTED」で後続 commit 率を比べる。
人間の CHANGES_REQUESTED(定義上「直してください」)は上限の参照値として併記する。

交絡: レビューが PR ライフサイクルのどこで出たかで後続 commit 率は当然変わる
(早い段階のレビューほど後に commit が来やすい)。そこで
  - PR 作成からの経過日数
  - PR 内で何番目のレビューか
  - PR の規模
を統制した線形確率モデルも推定する。さらに **同一 PR 内に AI と人間の COMMENTED が
両方ある部分標本で PR 固定効果**を入れ、PR の性質を完全に吸収した比較も行う。

出力: pilot/phase2/outputs/d13_review_level.parquet, d13_review_response.json
実行: uv run python pilot/phase2/scripts/d13_review_response.py
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, REPO_ROOT, load_repos, repo_dir_name

# RAW_DIR は common 側で環境変数 PILOT_RAW により切り替わる
WINDOWS = [1, 7]  # 応答とみなす日数


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def build_for_repo(repo: str) -> pd.DataFrame | None:
    interim_d = INTERIM_DIR / repo_dir_name(repo)
    prs_path = RAW_DIR / repo_dir_name(repo) / "prs_full.jsonl"
    actors_path = interim_d / "actors.parquet"
    events_path = interim_d / "events.parquet"
    if not (prs_path.exists() and actors_path.exists() and events_path.exists()):
        return None

    actors = pd.read_parquet(actors_path)
    cat_of = dict(zip(actors["actor_login"], actors["category"]))
    g = pd.read_parquet(events_path)
    g = g[g["stage"] == "G"]
    pr_to_module = dict(zip(g["pr_number"], g["module_id"]))

    rows = []
    for pr in read_jsonl(prs_path):
        number = pr["number"]
        module_id = pr_to_module.get(number)
        if module_id is None or not pr.get("createdAt"):
            continue
        created = parse_dt(pr["createdAt"])

        commit_times = []
        for c in (pr.get("commits") or {}).get("nodes", []):
            cm = c.get("commit") or {}
            if cm.get("committedDate"):
                commit_times.append(parse_dt(cm["committedDate"]))
        commit_times.sort()

        reviews = []
        for rv in (pr.get("reviews") or {}).get("nodes", []):
            if not rv.get("submittedAt") or not rv.get("author"):
                continue
            reviews.append(
                (parse_dt(rv["submittedAt"]), cat_of.get(rv["author"]["login"], "unknown"), rv.get("state"))
            )
        reviews.sort(key=lambda x: x[0])
        if not reviews:
            continue

        merged_at = parse_dt(pr["mergedAt"]) if pr.get("mergedAt") else None
        pr_files = (pr.get("files") or {}).get("nodes", [])
        pr_lines = sum(f.get("additions", 0) + f.get("deletions", 0) for f in pr_files)
        n_commits_total = len(commit_times)

        for idx, (t, cat, state) in enumerate(reviews):
            if cat not in ("human", "agent"):
                continue
            after = [c for c in commit_times if c > t]
            rec = {
                "repo": repo,
                "module_id_global": f"{repo}::{module_id}",
                "month": pr["createdAt"][:7],
                "pr_number": number,
                "review_idx": idx,
                "category": cat,
                "state": state,
                "days_since_pr_created": (t - created).total_seconds() / 86400,
                "n_commits_after": len(after),
                "is_last_review": idx == len(reviews) - 1,
                "n_reviews_in_pr": len(reviews),
                "n_commits_total": n_commits_total,
                "pr_lines": pr_lines,
                "pr_files": len(pr_files),
                "merged": merged_at is not None,
            }
            for w in WINDOWS:
                rec[f"commit_within_{w}d"] = any(c <= t + timedelta(days=w) for c in after)
            # PR がまだ開いている/レビュー後すぐ閉じた場合の打ち切りを把握するため
            rec["pr_open_days_after_review"] = (
                ((merged_at - t).total_seconds() / 86400) if merged_at else np.nan
            )
            rows.append(rec)
    if not rows:
        return None
    return pd.DataFrame(rows)


def main() -> None:
    frames = []
    for repo in load_repos():
        df = build_for_repo(repo)
        if df is not None:
            frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_parquet(OUT_DIR / "d13_review_level.parquet", index=False)

    out: dict = {"n_reviews": int(len(all_df)), "n_repos": int(all_df["repo"].nunique())}
    print(f"レビュー単位データ: {len(all_df):,} 件, {all_df['repo'].nunique()} リポジトリ\n")

    # --- 1. 記述: 後続 commit 率 ------------------------------------------------
    print("=== 後続 commit 率(レビュー種別 x 担い手) ===")
    grp = all_df.groupby(["category", "state"]).agg(
        n=("pr_number", "size"),
        commit_1d=("commit_within_1d", "mean"),
        commit_7d=("commit_within_7d", "mean"),
        any_after=("n_commits_after", lambda s: (s > 0).mean()),
        mean_commits_after=("n_commits_after", "mean"),
    )
    grp = grp[grp["n"] >= 50].sort_values("n", ascending=False)
    print(f"{'担い手':<8}{'state':<20}{'n':>9}{'1日以内':>9}{'7日以内':>9}{'以降いつか':>10}{'平均commit数':>12}")
    for (cat, state), r in grp.iterrows():
        print(f"{cat:<8}{state:<20}{int(r['n']):>9,}{r['commit_1d']:>9.1%}{r['commit_7d']:>9.1%}"
              f"{r['any_after']:>10.1%}{r['mean_commits_after']:>12.2f}")
    out["descriptive"] = grp.reset_index().to_dict(orient="records")

    # --- 2. COMMENTED だけに絞った公平な比較 --------------------------------------
    com = all_df[all_df["state"] == "COMMENTED"].copy()
    com["is_agent"] = (com["category"] == "agent").astype(float)
    com["log_days_since"] = np.log1p(com["days_since_pr_created"].clip(lower=0))
    com["log_pr_lines"] = np.log1p(com["pr_lines"])
    com["log_review_idx"] = np.log1p(com["review_idx"])

    print("\n=== COMMENTED に限定した比較(同じ種別のレビューで比べる) ===")
    for cat in ["human", "agent"]:
        s = com[com["category"] == cat]
        print(f"  {cat:<6} n={len(s):>8,}  7日以内commit {s['commit_within_7d'].mean():.1%}  "
              f"以降いつか {(s['n_commits_after'] > 0).mean():.1%}")

    # 線形確率モデル: PR ライフサイクル上の位置と規模を統制
    X_cols = ["is_agent", "log_days_since", "log_review_idx", "log_pr_lines"]
    d = com.dropna(subset=X_cols + ["commit_within_7d"])
    X = sm.add_constant(d[X_cols])
    y = d["commit_within_7d"].astype(float)
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["repo"]})
    out["lpm_controls"] = {
        "params": res.params.to_dict(), "pvalues": res.pvalues.to_dict(), "n": int(res.nobs),
    }
    print(f"\n  統制付き線形確率モデル(repoクラスタ, n={int(res.nobs):,}):")
    print(f"    is_agent = {res.params['is_agent']:+.4f}  (p={res.pvalues['is_agent']:.4g})")

    # --- 3. 同一 PR 内比較(PR 固定効果) ------------------------------------------
    # AI と人間の COMMENTED が両方ある PR だけを使い、PR の性質を完全に吸収する
    has_both = com.groupby(["repo", "pr_number"])["category"].nunique()
    both_keys = set(has_both[has_both == 2].index)
    sub = com[com.set_index(["repo", "pr_number"]).index.isin(both_keys)].copy()
    print(f"\n=== 同一 PR 内比較(AI と人間の COMMENTED が両方ある PR) ===")
    print(f"  対象 PR 数: {len(both_keys):,}  レビュー数: {len(sub):,}")
    if len(sub) > 100:
        sub["pr_key"] = sub["repo"] + "#" + sub["pr_number"].astype(str)
        sub["obs"] = np.arange(len(sub))
        dd = sub.set_index(["pr_key", "obs"])
        exog = dd[["is_agent", "log_days_since", "log_review_idx"]]
        r2 = PanelOLS(dd["commit_within_7d"].astype(float), exog,
                      entity_effects=True, drop_absorbed=True).fit(
            cov_type="clustered", clusters=dd["repo"])
        out["pr_fixed_effects"] = {
            "params": r2.params.to_dict(), "pvalues": r2.pvalues.to_dict(), "n": int(r2.nobs),
            "n_prs": len(both_keys),
        }
        print(f"  PR 固定効果モデル(repoクラスタ, n={int(r2.nobs):,}):")
        print(f"    is_agent = {r2.params['is_agent']:+.4f}  (p={r2.pvalues['is_agent']:.4g})")
        for c in ["log_days_since", "log_review_idx"]:
            print(f"    {c:<16} = {r2.params[c]:+.4f}  (p={r2.pvalues[c]:.4g})")

    # --- 4. AI レビューだけが付いてマージされた PR ---------------------------------
    pr_lvl = all_df.groupby(["repo", "pr_number"]).agg(
        n_ai=("category", lambda s: (s == "agent").sum()),
        n_human=("category", lambda s: (s == "human").sum()),
        merged=("merged", "first"),
    ).reset_index()
    ai_only = pr_lvl[(pr_lvl["n_ai"] > 0) & (pr_lvl["n_human"] == 0)]
    out["ai_only_prs"] = {
        "n": int(len(ai_only)),
        "share_of_reviewed_prs": float(len(ai_only) / len(pr_lvl)),
        "merged_rate": float(ai_only["merged"].mean()) if len(ai_only) else None,
    }
    print(f"\n=== AI しかレビューしていない PR ===")
    print(f"  {len(ai_only):,} 件(レビューがある PR の {len(ai_only)/len(pr_lvl):.1%})  "
          f"マージ率 {ai_only['merged'].mean():.1%}")

    (OUT_DIR / "d13_review_response.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd13_review_response.json'}")


if __name__ == "__main__":
    main()

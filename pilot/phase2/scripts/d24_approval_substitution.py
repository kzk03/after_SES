"""D24: ★判断権限が代替の鍵か — ボットの承認は人間の承認を省略させるか。

phase2 の中心的な主張は「AI は判断(APPROVED / CHANGES_REQUESTED)に参加していないので、
人間のレビュー作業を代替しない」である。しかしこれは相関的な説明にとどまっている。
「判断すれば代替が起きる」ことを示せれば、主張は因果的なメカニズムに近づく。

手元には対照群がある。**legacy_bot(CI ボット等)は判断を出している**:
  github-actions       11,160件中 APPROVED 8,153件(判断率 73.6%)
  home-assistant[bot]   5,599件中 CHANGES_REQUESTED 1,845件(33.0%)
  grafana-pr-approver     315件中 APPROVED 314件(99.7%)
対して AI エージェントの判断率は 0.27%。

ただし legacy_bot の「判断」は CI の結果報告であって、設計の妥当性を見る人間のレビューとは
タスクが違う。そこで「判断するボットは人間を代替するか」と大雑把に問うのではなく、
**「ボットが APPROVED を出した PR で、人間の APPROVED は省略されるか」**に絞る。

  - 省略される  → 承認という判断は、権限を持つ主体なら機械でも代替できる
                  = AI が代替しないのは技術ではなく**権限が与えられていないから**
  - 省略されない → 人間の承認は制度上必須で、誰も代替できない(別の説明が必要)

交絡: ボットが承認する PR は CI が通る簡単な PR で、人間の承認も得やすいかもしれない。
そこで PR 規模を統制し、さらに **module 固定効果 + 月固定効果**で「同じモジュール・同じ月の中での差」を見る。

★ **最初の実行で決定的な交絡が見つかった(2026-09-22)。**
legacy_bot が承認した 5,000 PR のうち **4,965件(99.3%)は legacy_bot 自身が作成した PR**
(dependabot 4,653件、grafana-pr-automation 312件)だった。
「ボットが依存更新 PR を作り、CI ボットが自動承認する」という自己完結したループで、
人間の承認が省略されるのは当然。**PR 作成者のカテゴリで必ず層別すること。**
人間が作成した PR に限ると legacy_bot の承認はわずか 35件しかない。

出力: pilot/phase2/outputs_full/d24_approval.parquet, d24_approval_substitution.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d24_approval_substitution.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name


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


def build_for_repo(repo: str) -> pd.DataFrame | None:
    interim_d = INTERIM_DIR / repo_dir_name(repo)
    prs_path = RAW_DIR / repo_dir_name(repo) / "prs_full.jsonl"
    actors_path = interim_d / "actors.parquet"
    events_path = interim_d / "events.parquet"
    if not (prs_path.exists() and actors_path.exists() and events_path.exists()):
        return None

    a = pd.read_parquet(actors_path)
    cat_of = dict(zip(a["actor_login"], a["category"]))
    g = pd.read_parquet(events_path)
    pr_to_module = dict(zip(g[g["stage"] == "G"]["pr_number"], g[g["stage"] == "G"]["module_id"]))

    rows = []
    for pr in read_jsonl(prs_path):
        module_id = pr_to_module.get(pr["number"])
        if module_id is None or not pr.get("createdAt"):
            continue
        counts = {c: {"APPROVED": 0, "CHANGES_REQUESTED": 0, "COMMENTED": 0}
                  for c in ("human", "agent", "legacy_bot")}
        # 「最初に APPROVED を出したのは誰か」も取る(順序が分かると省略の議論がしやすい)
        first_approver_cat = None
        first_approved_at = None
        for rv in (pr.get("reviews") or {}).get("nodes", []):
            if not rv.get("author") or not rv.get("submittedAt"):
                continue
            cat = cat_of.get(rv["author"]["login"])
            if cat not in counts:
                continue
            st = rv.get("state")
            if st in counts[cat]:
                counts[cat][st] += 1
            if st == "APPROVED":
                t = datetime.fromisoformat(rv["submittedAt"].replace("Z", "+00:00"))
                if first_approved_at is None or t < first_approved_at:
                    first_approved_at, first_approver_cat = t, cat

        author = (pr.get("author") or {}).get("login")
        files = (pr.get("files") or {}).get("nodes", [])
        rows.append({
            "repo": repo,
            "module_id_global": f"{repo}::{module_id}",
            "month": pr["createdAt"][:7],
            "pr_number": pr["number"],
            "author_login": author,
            "author_cat": cat_of.get(author, "unknown"),
            "human_appr": counts["human"]["APPROVED"],
            "agent_appr": counts["agent"]["APPROVED"],
            "legacy_appr": counts["legacy_bot"]["APPROVED"],
            "legacy_cr": counts["legacy_bot"]["CHANGES_REQUESTED"],
            "human_reviews": sum(counts["human"].values()),
            "first_approver_cat": first_approver_cat,
            "merged": pr.get("mergedAt") is not None,
            "pr_lines": sum(f.get("additions", 0) + f.get("deletions", 0) for f in files),
            "pr_files": len(files),
        })
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    frames = []
    for repo in load_repos():
        df = build_for_repo(repo)
        if df is not None:
            frames.append(df)
    pr = pd.concat(frames, ignore_index=True)
    pr["has_human_appr"] = (pr["human_appr"] > 0).astype(float)
    pr["has_agent_appr"] = (pr["agent_appr"] > 0).astype(float)
    pr["has_legacy_appr"] = (pr["legacy_appr"] > 0).astype(float)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pr.to_parquet(OUT_DIR / "d24_approval.parquet", index=False)

    out: dict = {"n_prs": int(len(pr)), "n_repos": int(pr["repo"].nunique())}
    print(f"=== 全 PR {len(pr):,} 件({pr['repo'].nunique()} リポジトリ) ===\n")

    # --- 0. ★交絡の確認: ボットの承認は誰の PR に向いているか ----------------------
    la = pr[pr["has_legacy_appr"] == 1]
    print("【★交絡の確認】legacy_bot が承認した PR の作成者")
    vc = la["author_cat"].value_counts()
    for k, v in vc.items():
        print(f"  {k:<14}{v:>8,}  ({v / len(la):>6.1%})")
    print(f"  作成者 login 上位: "
          + ", ".join(f"{k}({v:,})" for k, v in la["author_login"].value_counts().head(4).items()))
    out["legacy_approved_pr_authors"] = {str(k): int(v) for k, v in vc.items()}
    print("  → ボットの承認はほぼボット自身の PR に向いている。**人間作成 PR に限定して分析する**\n")

    # --- 1. 記述: 人間が作成した PR に限定して、ボット承認の有無で比べる -------------
    hp = pr[pr["author_cat"] == "human"].copy()
    print(f"【記述】人間が作成した PR {len(hp):,} 件に限定")
    print(f"  {'条件':<34}{'PR数':>10}{'人間APPR率':>12}{'マージ率':>10}{'人間レビュー数':>14}")
    conds = [
        ("どのボットも承認なし", (hp["has_legacy_appr"] == 0) & (hp["has_agent_appr"] == 0)),
        ("legacy_bot が承認", hp["has_legacy_appr"] == 1),
        ("AI エージェントが承認", hp["has_agent_appr"] == 1),
    ]
    desc = {}
    for label, m in conds:
        s = hp[m]
        if len(s) == 0:
            continue
        desc[label] = {"n": int(len(s)), "human_appr_rate": float(s["has_human_appr"].mean()),
                       "merge_rate": float(s["merged"].mean()),
                       "human_reviews": float(s["human_reviews"].mean())}
        print(f"  {label:<34}{len(s):>10,}{s['has_human_appr'].mean():>12.1%}"
              f"{s['merged'].mean():>10.1%}{s['human_reviews'].mean():>14.2f}")
    out["descriptive_human_authored"] = desc

    # --- 2. マージされた PR のうち、人間の承認なしでマージされた割合 -----------------
    mg = hp[hp["merged"]]
    print(f"\n【人間作成・マージ済み {len(mg):,} 件】人間の APPROVED なしでマージされた割合")
    for label, m in conds:
        s = mg[m.reindex(mg.index, fill_value=False)]
        if len(s) == 0:
            continue
        print(f"  {label:<34}{len(s):>10,}{1 - s['has_human_appr'].mean():>12.1%}")

    # --- 3. 最初の承認者は誰か(人間作成 PR に限定) ---------------------------------
    fa = hp[hp["first_approver_cat"].notna()]["first_approver_cat"].value_counts()
    print(f"\n【人間作成 PR で最初に APPROVED を出した主体】({int(fa.sum()):,} 件)")
    for k, v in fa.items():
        print(f"  {k:<12}{v:>10,}  ({v / fa.sum():>6.1%})")
    out["first_approver_human_authored"] = {k: int(v) for k, v in fa.items()}

    # --- 4. 固定効果モデル: 同じモジュール・同じ月の中での比較 -----------------------
    # 「ボットが承認した PR では人間の承認が省略されるか」を PR 規模を統制して推定する
    hp["log_lines"] = np.log1p(hp["pr_lines"])
    hp["log_files"] = np.log1p(hp["pr_files"])
    hp["obs"] = np.arange(len(hp))
    d = hp.dropna(subset=["log_lines", "log_files"]).copy()
    # entity = module、time は月ダミーで入れる(1 module x 月に複数 PR があるため time_effects は使えない)
    dummies = pd.get_dummies(d["month"], drop_first=True, dtype=float)

    print("\n【固定効果モデル】人間作成 PR に限定。被説明変数 = 人間の APPROVED があったか(module FE + 月ダミー, repoクラスタ)")
    out["models"] = {}
    for label, cols in [
        ("legacy_bot の承認", ["has_legacy_appr", "log_lines", "log_files"]),
        ("AI エージェントの承認", ["has_agent_appr", "log_lines", "log_files"]),
        ("両方同時に", ["has_legacy_appr", "has_agent_appr", "log_lines", "log_files"]),
    ]:
        X = pd.concat([d[cols], dummies], axis=1)
        dd = pd.concat([d[["module_id_global", "obs", "has_human_appr", "repo"]], X], axis=1)
        dd = dd.set_index(["module_id_global", "obs"])
        try:
            res = PanelOLS(dd["has_human_appr"], dd[X.columns], entity_effects=True,
                           drop_absorbed=True).fit(cov_type="clustered", clusters=dd["repo"])
        except Exception as e:
            print(f"  [{label}] 推定できず ({type(e).__name__})")
            continue
        rec = {}
        print(f"  [{label}] n={int(res.nobs):,}")
        for c in cols:
            if c not in res.params:
                continue
            rec[c] = {"coef": float(res.params[c]), "se": float(res.std_errors[c]),
                      "p": float(res.pvalues[c])}
            star = " *" if res.pvalues[c] < 0.05 else ""
            print(f"    {c:<20} {res.params[c]:+.4f} (se={res.std_errors[c]:.4f}, "
                  f"p={res.pvalues[c]:.4g}){star}")
        out["models"][label] = rec

    (OUT_DIR / "d24_approval_substitution.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'd24_approval_substitution.json'}")


if __name__ == "__main__":
    main()

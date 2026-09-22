"""D27: AI はどこから進出し、人間の振る舞いはどう変わったか。

ここまでで「人間のレビュー**量**は減らない」ことは分かった(D4, D9)。
しかし量が同じでも**振る舞い**が変わっている可能性がある。量的代替がなくても行動変容はありうる。

2つを見る。

【Part A】AI はどういうモジュールから進出したか
  AI が評価に入るモジュールは無作為ではないはず。
  「人手が足りない(レビューが薄い)ところ」から入るのか、
  「もともと活発なところ」から入るのか。参入順序とモジュールの事前特性を突き合わせる。

【Part B】人間の振る舞いはどう変わったか
  D21 で「同じ箇所では 78.6% で AI が先」と分かった。では導入後、
  **人間はレビューを後ろ倒しにした(AI を待つようになった)のか**。
  - PR 作成から人間の最初のレビューまでの時間
  - 人間のレビューが AI より後に来る割合
  - PR あたりの人間レビュアー数
  をリポジトリの AI レビュー導入イベントで推定する(D4 と同じ設計)。

実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PILOT_PROCESSED=processed_full \
      PHASE2_OUT=outputs_full uv run python pilot/phase2/scripts/d27_behavior_shift.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, build_count_panel, load_repos, repo_dir_name
from d4_repo_event_study import LEAD_LAG, REF, detect_adoption


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


def build_pr_timing() -> pd.DataFrame:
    """PR 単位で「誰がいつレビューしたか」のタイミング指標を作る。"""
    rows = []
    for repo in load_repos():
        prs_path = RAW_DIR / repo_dir_name(repo) / "prs_full.jsonl"
        actors_path = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        events_path = INTERIM_DIR / repo_dir_name(repo) / "events.parquet"
        if not (prs_path.exists() and actors_path.exists() and events_path.exists()):
            continue
        a = pd.read_parquet(actors_path)
        cat_of = dict(zip(a["actor_login"], a["category"]))
        g = pd.read_parquet(events_path)
        g = g[g["stage"] == "G"]
        pr_to_module = dict(zip(g["pr_number"], g["module_id"]))

        for pr in read_jsonl(prs_path):
            module_id = pr_to_module.get(pr["number"])
            if module_id is None or not pr.get("createdAt"):
                continue
            created = parse_dt(pr["createdAt"])
            first = {"human": None, "agent": None}
            human_reviewers = set()
            for rv in (pr.get("reviews") or {}).get("nodes", []):
                if not rv.get("author") or not rv.get("submittedAt"):
                    continue
                login = rv["author"]["login"]
                cat = cat_of.get(login)
                if cat not in ("human", "agent"):
                    continue
                t = parse_dt(rv["submittedAt"])
                if first[cat] is None or t < first[cat]:
                    first[cat] = t
                if cat == "human":
                    human_reviewers.add(login)
            if first["human"] is None:
                continue  # 人間のレビューが無い PR はタイミングを定義できない
            lag_h = (first["human"] - created).total_seconds() / 3600  # 時間
            if lag_h < 0:
                continue
            rows.append({
                "repo": repo,
                "module_id_global": f"{repo}::{module_id}",
                "month": pr["createdAt"][:7],
                "pr_number": pr["number"],
                "human_first_lag_h": lag_h,
                "ai_reviewed": 1.0 if first["agent"] is not None else 0.0,
                # AI もレビューした PR で、人間が AI より後に来たか
                "human_after_ai": (1.0 if (first["agent"] is not None and first["human"] > first["agent"])
                                   else (0.0 if first["agent"] is not None else np.nan)),
                "n_human_reviewers": len(human_reviewers),
            })
    return pd.DataFrame(rows)


def fit(d: pd.DataFrame, y: str, cols: list[str], entity: str = "module_id_global") -> dict:
    dd = d.dropna(subset=[y, *cols]).set_index([entity, "month_period"])
    if dd.empty or dd["repo"].nunique() < 5:
        return {}
    try:
        r = PanelOLS(dd[y], dd[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
            cov_type="clustered", clusters=dd["repo"])
    except ValueError:
        return {}
    return {"params": r.params.to_dict(), "p": r.pvalues.to_dict(),
            "se": r.std_errors.to_dict(), "n": int(r.nobs)}


def show(r: dict, label: str) -> None:
    if not r:
        print(f"  {label:<30} (推定できず)")
        return
    m2, pm2 = r["params"].get("k_m2", np.nan), r["p"].get("k_m2", np.nan)
    b2, p2 = r["params"].get("k_p2", np.nan), r["p"].get("k_p2", np.nan)
    b3, p3 = r["params"].get("k_p3", np.nan), r["p"].get("k_p3", np.nan)
    star = " *" if (p2 < 0.05 or p3 < 0.05) else ""
    print(f"  {label:<30}{m2:>+10.3f}{pm2:>8.3f}{b2:>+10.3f}{p2:>8.3f}{b3:>+10.3f}{p3:>8.3f}"
          f"{r['n']:>9,}{star}")


def main() -> None:
    df = build_count_panel()
    months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(months)}
    adoption, never, always = detect_adoption(df)
    out: dict = {}

    # =====================================================================
    # Part A: AI はどういうモジュールから進出したか
    # =====================================================================
    print("=" * 76)
    print("【Part A】AI はどういうモジュールから進出したか")
    print("=" * 76)

    # 各モジュールについて「AI 評価が初めて入った月」と、その直前までの特性を取る
    df = df.sort_values(["module_id_global", "month"])
    df["has_agent_E"] = df["n_E_agent"] > 0
    first_ai = df[df["has_agent_E"]].groupby("module_id_global")["month"].min().rename("first_ai_month")
    d = df.join(first_ai, on="module_id_global")
    # AI 参入前の期間だけで、モジュールの事前特性を作る
    pre = d[(d["first_ai_month"].isna()) | (d["month"] < d["first_ai_month"])]
    feat = pre.groupby(["module_id_global", "repo"]).agg(
        n_months_pre=("month", "nunique"),
        avg_E=("n_E_human", "mean"),
        avg_G=("n_G", "mean"),
        avg_reviewers=("n_E_human_actors", "mean"),
        avg_prs=("n_G_prs", "mean"),
    ).reset_index()
    # レビュー密度 = PR あたりの人間レビュー数(低い = 人手が足りていない)
    feat["review_density"] = feat["avg_E"] / feat["avg_prs"].replace(0, np.nan)
    feat = feat.join(first_ai, on="module_id_global")
    feat["ai_entered"] = feat["first_ai_month"].notna().astype(float)
    feat = feat[feat["n_months_pre"] >= 3]

    print(f"\nAI 参入前の期間が3ヶ月以上あるモジュール: {len(feat):,}")
    print(f"  うち AI が参入した: {int(feat['ai_entered'].sum()):,} ({feat['ai_entered'].mean():.1%})")

    print("\n[AI 参入の有無で、事前特性を比べる]")
    print(f"  {'指標':<26}{'AI参入あり':>14}{'AI参入なし':>14}")
    for col, lab in [("avg_prs", "月あたり PR 数"), ("avg_E", "月あたり人間レビュー数"),
                     ("avg_reviewers", "月あたりレビュアー数"), ("review_density", "PRあたり人間レビュー数")]:
        a1 = feat.loc[feat["ai_entered"] == 1, col].median()
        a0 = feat.loc[feat["ai_entered"] == 0, col].median()
        print(f"  {lab:<26}{a1:>14.2f}{a0:>14.2f}")

    # どういうモジュールに AI が入るか(repo 固定効果込みの線形確率モデル)
    fm = feat.dropna(subset=["review_density"]).copy()
    for c in ["avg_prs", "avg_E", "avg_reviewers", "review_density"]:
        fm[f"log_{c}"] = np.log1p(fm[c])
    X_cols = ["log_avg_prs", "log_avg_reviewers", "log_review_density"]
    dummies = pd.get_dummies(fm["repo"], drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([fm[X_cols], dummies], axis=1))
    res = sm.OLS(fm["ai_entered"], X).fit(cov_type="cluster", cov_kwds={"groups": fm["repo"]})
    out["part_a"] = {"params": {c: float(res.params[c]) for c in X_cols},
                     "p": {c: float(res.pvalues[c]) for c in X_cols}, "n": int(res.nobs)}
    print(f"\n[線形確率モデル] 被説明変数 = AI が参入したか (repo 固定効果, repoクラスタ, n={int(res.nobs):,})")
    for c in X_cols:
        star = " *" if res.pvalues[c] < 0.05 else ""
        print(f"  {c:<24}{res.params[c]:+.4f} (p={res.pvalues[c]:.4g}){star}")
    print("  → 係数が正 = その特性が高いモジュールほど AI が入りやすい")

    # =====================================================================
    # Part B: 人間の振る舞いはどう変わったか
    # =====================================================================
    print("\n" + "=" * 76)
    print("【Part B】人間の振る舞いはどう変わったか(処置 = リポジトリの AI レビュー導入)")
    print("=" * 76)

    pr = build_pr_timing()
    print(f"\n人間のレビューがある PR {len(pr):,} 件")
    print(f"  うち AI もレビューした PR: {int(pr['ai_reviewed'].sum()):,}")
    sub = pr[pr["human_after_ai"].notna()]
    print(f"  AI もレビューした PR のうち、人間が AI より後に来た割合: {sub['human_after_ai'].mean():.1%}")

    agg = pr.groupby(["repo", "module_id_global", "month"]).agg(
        human_first_lag_h=("human_first_lag_h", "median"),
        human_after_ai=("human_after_ai", "mean"),
        n_human_reviewers=("n_human_reviewers", "mean"),
        n_prs=("pr_number", "size"),
    ).reset_index()
    agg = agg[agg["repo"].isin(set(adoption) | set(never))].copy()
    agg["month_period"] = pd.PeriodIndex(agg["month"], freq="M").to_timestamp()
    agg["month_idx"] = agg["month"].map(midx)
    agg = agg.dropna(subset=["month_idx"])
    agg["g"] = agg["repo"].map({r: midx[m] for r, m in adoption.items()})
    k = agg["month_idx"] - agg["g"]
    ev = []
    for j in range(-LEAD_LAG, LEAD_LAG + 1):
        if j == REF:
            continue
        nm = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
        val = (k <= j) if j == -LEAD_LAG else ((k >= j) if j == LEAD_LAG else (k == j))
        agg[nm] = val.fillna(False).astype(float)
        ev.append(nm)
    agg["log_lag"] = np.log1p(agg["human_first_lag_h"])
    agg["log_reviewers"] = np.log1p(agg["n_human_reviewers"])

    print(f"\n  {'被説明変数':<30}{'k=-2':>10}{'p':>8}{'k=+2':>10}{'p':>8}{'k=+3':>10}{'p':>8}{'n':>9}")
    out["part_b"] = {}
    for y, lab in [("log_lag", "log(1+人間の初回レビュー遅れ時間)"),
                   ("human_after_ai", "人間が AI より後に来る割合"),
                   ("log_reviewers", "log(1+PRあたり人間レビュアー数)")]:
        r = fit(agg, y, ev)
        out["part_b"][y] = r
        show(r, lab)

    (OUT_DIR / "d27_behavior_shift.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'd27_behavior_shift.json'}")


if __name__ == "__main__":
    main()

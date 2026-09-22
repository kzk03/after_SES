"""D25: 個人内で審査は甘くなっているか — Habituation at the Gate との直接対決。

[Habituation at the Gate](https://arxiv.org/abs/2606.22721) は AIDev で 400人 × 7ヶ月の
レビュアー個人内変化を追い、「**承認率が上がりコメントが減る = 馴化による審査の劣化**」を報告している。

これは本研究の結果と正面から食い違って見える。我々は集団レベルで
「人間のレビュー件数・人数・PR あたり関与は減らない」(D4)、
「人間の CHANGES_REQUESTED も APPROVED も変化しない」(D10)と言っている。

食い違いの候補:
  - 粒度: 彼らは個人内、我々はモジュール集計。**個人が薄めても新規参入で埋まれば集計は動かない**
  - 期間: 彼らは7ヶ月、我々は45ヶ月
  - 対象: 彼らは AI-PR のレビュアー、我々は全 PR

そこで本スクリプトでは **レビュアー個人 × 月**のパネルを作り、彼らと同じ土俵(個人固定効果)で
承認率・1レビューあたりコメント数・レビュー件数を見る。
さらに **PR 作成者が AI か人間か**で分けて推定し、「AI-PR に対してだけ甘くなる」のかも確かめる。

処置は D4 と同じ「リポジトリの AI レビューボット導入」。

実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PILOT_PROCESSED=processed_full \
      PHASE2_OUT=outputs_full uv run python pilot/phase2/scripts/d25_habituation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, build_count_panel, load_repos, repo_dir_name
from d4_repo_event_study import LEAD_LAG, REF, detect_adoption

MIN_MONTHS = 3      # 個人固定効果を使うので、複数月観測できる人に限る
MIN_REVIEWS = 2     # 月あたり最低レビュー数(比率の分母が小さすぎると不安定)


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


def build_reviewer_month() -> pd.DataFrame:
    """レビュアー個人 x 月 x (PR作成者カテゴリ) のパネルを作る。"""
    rows = []
    for repo in load_repos():
        prs_path = RAW_DIR / repo_dir_name(repo) / "prs_full.jsonl"
        actors_path = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not (prs_path.exists() and actors_path.exists()):
            continue
        a = pd.read_parquet(actors_path)
        cat_of = dict(zip(a["actor_login"], a["category"]))

        for pr in read_jsonl(prs_path):
            author = (pr.get("author") or {}).get("login")
            author_cat = cat_of.get(author, "unknown")
            for rv in (pr.get("reviews") or {}).get("nodes", []):
                if not rv.get("author") or not rv.get("submittedAt"):
                    continue
                login = rv["author"]["login"]
                if cat_of.get(login) != "human":
                    continue  # 人間レビュアーの行動だけを見る
                rows.append({
                    "repo": repo,
                    "actor": f"{repo}::{login}",
                    "month": rv["submittedAt"][:7],
                    "author_is_agent": 1 if author_cat == "agent" else 0,
                    "approved": 1 if rv.get("state") == "APPROVED" else 0,
                    "changes_requested": 1 if rv.get("state") == "CHANGES_REQUESTED" else 0,
                })
    return pd.DataFrame(rows)


def aggregate(rv: pd.DataFrame, subset: str) -> pd.DataFrame:
    """subset: all / ai_pr / human_pr"""
    d = rv
    if subset == "ai_pr":
        d = rv[rv["author_is_agent"] == 1]
    elif subset == "human_pr":
        d = rv[rv["author_is_agent"] == 0]
    g = d.groupby(["repo", "actor", "month"]).agg(
        n_reviews=("approved", "size"),
        approval_rate=("approved", "mean"),
        cr_rate=("changes_requested", "mean"),
    ).reset_index()
    g = g[g["n_reviews"] >= MIN_REVIEWS]
    keep = g.groupby("actor")["month"].transform("nunique") >= MIN_MONTHS
    return g[keep].copy()


def fit(d: pd.DataFrame, y: str, cols: list[str]) -> dict:
    dd = d.dropna(subset=[y, *cols]).set_index(["actor", "month_period"])
    if dd.empty or dd["repo"].nunique() < 5:
        return {}
    try:
        r = PanelOLS(dd[y], dd[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
            cov_type="clustered", clusters=dd["repo"])
    except ValueError:
        # サンプルが薄いとイベントダミーが固定効果に飲まれてランク落ちする
        return {}
    return {"params": r.params.to_dict(), "se": r.std_errors.to_dict(),
            "p": r.pvalues.to_dict(), "n": int(r.nobs),
            "n_actors": int(dd.index.get_level_values(0).nunique())}


def main() -> None:
    df = build_count_panel()
    months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(months)}
    adoption, never, always = detect_adoption(df)

    rv = build_reviewer_month()
    print(f"人間レビュアーのレビュー {len(rv):,} 件 "
          f"(うち AI 作成 PR へのレビュー {rv['author_is_agent'].sum():,} 件)\n")

    out: dict = {"adoption": adoption, "models": {}}
    for subset, label in [("all", "全 PR"), ("human_pr", "人間作成 PR のみ"), ("ai_pr", "AI 作成 PR のみ")]:
        g = aggregate(rv, subset)
        g = g[g["repo"].isin(set(adoption) | set(never))].copy()
        if len(g) < 500:
            print(f"=== {label}: サンプル不足({len(g)}) ===\n")
            continue
        g["month_period"] = pd.PeriodIndex(g["month"], freq="M").to_timestamp()
        g["month_idx"] = g["month"].map(midx)
        g["g"] = g["repo"].map({r: midx[m] for r, m in adoption.items()})
        k = g["month_idx"] - g["g"]
        ev = []
        for j in range(-LEAD_LAG, LEAD_LAG + 1):
            if j == REF:
                continue
            nm = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
            val = (k <= j) if j == -LEAD_LAG else ((k >= j) if j == LEAD_LAG else (k == j))
            g[nm] = val.fillna(False).astype(float)
            ev.append(nm)
        g["log_n_reviews"] = np.log1p(g["n_reviews"])

        print(f"=== {label} (レビュアー {g['actor'].nunique():,}人, 観測 {len(g):,}) ===")
        print(f"  {'被説明変数':<20}{'k=-2':>10}{'p':>8}{'k=+2':>10}{'p':>8}{'k=+3':>10}{'p':>8}")
        out["models"][label] = {}
        for y in ["approval_rate", "cr_rate", "log_n_reviews"]:
            r = fit(g, y, ev)
            if not r:
                print(f"  {y:<20} (推定できず)")
                continue
            out["models"][label][y] = r
            m2, pm2 = r["params"].get("k_m2", np.nan), r["p"].get("k_m2", np.nan)
            b2, p2 = r["params"].get("k_p2", np.nan), r["p"].get("k_p2", np.nan)
            b3, p3 = r["params"].get("k_p3", np.nan), r["p"].get("k_p3", np.nan)
            star = " *" if (p2 < 0.05 or p3 < 0.05) else ""
            print(f"  {y:<20}{m2:>+10.4f}{pm2:>8.3f}{b2:>+10.4f}{p2:>8.3f}{b3:>+10.4f}{p3:>8.3f}{star}")
        print()

    # 記述: 個人の承認率は時間とともに上がっているか(処置と無関係な全体トレンド)
    g_all = aggregate(rv, "all")
    tr = g_all.groupby("month").agg(approval_rate=("approval_rate", "mean"),
                                    cr_rate=("cr_rate", "mean"),
                                    n=("actor", "size")).reset_index()
    print("=== 参考: 個人平均の承認率の推移(全体、処置と無関係) ===")
    for _, r in tr.iloc[::6].iterrows():
        print(f"  {r['month']}  承認率 {r['approval_rate']:.3f}  CR率 {r['cr_rate']:.3f}  (n={int(r['n']):,})")
    out["trend"] = tr.to_dict(orient="records")

    (OUT_DIR / "d25_habituation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'd25_habituation.json'}")


if __name__ == "__main__":
    main()

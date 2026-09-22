"""D8: AI レビュー導入が人材フロー(新規参入・離脱)に与える影響。

RQ2 の相関分析では、分離度が低い(=AI 関与度が高い)モジュールほど新規参入が多く、
新規参入者は小さいタスクから入っていた(参入障壁低下説)。
D4 と同じ因果識別(リポジトリ単位の AI レビュー導入イベント)で再検証する。

定義:
  n_entrants(m,t)  : 月 t にそのモジュールで初めて活動した人間の数
                     (観測期間の頭は「全員が新規」に見えるので、先頭 BURN_IN ヶ月を除外)
  churn_rate(m,t)  : 月 t に活動していた人間のうち、t+1..t+K に一度も活動しなかった人の割合
                     (末尾 K ヶ月は計算できないので除外)
  n_active_humans  : 月 t に活動した人間の数
  entrant_pr_lines : その月の新規参入者による PR の平均変更行数(参入障壁の代理)

実行: uv run python pilot/phase2/scripts/d8_talent_flow.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from common import INTERIM_DIR, OUT_DIR, build_count_panel, load_repos, repo_dir_name
from d4_repo_event_study import LEAD_LAG, REF, detect_adoption

BURN_IN = 3   # 先頭何ヶ月を「初出現」判定から除くか
CHURN_K = 3   # 離脱判定の先読み月数


def build_flow_panel(months: list[str]) -> pd.DataFrame:
    midx = {m: i for i, m in enumerate(months)}
    rows = []
    for repo in load_repos():
        ev_path = INTERIM_DIR / repo_dir_name(repo) / "events.parquet"
        if not ev_path.exists():
            continue
        ev = pd.read_parquet(ev_path)
        ev = ev[ev["stage"].isin(["G", "E"]) & (ev["actor_category"] == "human")]
        if ev.empty:
            continue
        active: dict[tuple[str, str], set[str]] = {
            (mod, mon): set(g["actor_login"]) for (mod, mon), g in ev.groupby(["module_id", "month"])
        }
        # 新規参入者(module, month) -> set
        entrants: dict[tuple[str, str], set[str]] = {}
        for mod in ev["module_id"].unique():
            seen: set[str] = set()
            for t in months:
                cur = active.get((mod, t), set())
                if midx[t] >= BURN_IN:
                    entrants[(mod, t)] = cur - seen
                seen |= cur

        for mod in ev["module_id"].unique():
            for t in months:
                ti = midx[t]
                cur = active.get((mod, t), set())
                rec = {
                    "repo": repo,
                    "module_id_global": f"{repo}::{mod}",
                    "module_id": mod,
                    "month": t,
                    "n_active_humans": len(cur),
                    "n_entrants": len(entrants.get((mod, t), set())) if ti >= BURN_IN else np.nan,
                }
                # 離脱率: 末尾 CHURN_K ヶ月は先が見えないので NaN
                if cur and ti + CHURN_K < len(months):
                    future: set[str] = set()
                    for j in range(ti + 1, ti + 1 + CHURN_K):
                        future |= active.get((mod, months[j]), set())
                    rec["churn_rate"] = len(cur - future) / len(cur)
                else:
                    rec["churn_rate"] = np.nan
                rec["entrant_logins"] = entrants.get((mod, t), set()) if ti >= BURN_IN else set()
                rows.append(rec)
    return pd.DataFrame(rows)


def add_entrant_task_size(flow: pd.DataFrame) -> pd.DataFrame:
    """新規参入者がその月に出した PR の平均規模を付ける(参入障壁の代理)。"""
    pr = pd.read_parquet(OUT_DIR / "d5_pr_level.parquet")
    ev_author = []
    for repo in load_repos():
        ev_path = INTERIM_DIR / repo_dir_name(repo) / "events.parquet"
        if not ev_path.exists():
            continue
        ev = pd.read_parquet(ev_path)
        g = ev[(ev["stage"] == "G") & (ev["actor_category"] == "human")]
        if g.empty:
            continue
        ev_author.append(
            g[["pr_number", "actor_login", "module_id", "month"]].assign(repo=repo)
        )
    authors = pd.concat(ev_author, ignore_index=True)
    authors["module_id_global"] = authors["repo"] + "::" + authors["module_id"]
    pr_small = pr[["repo", "pr_number", "pr_lines", "pr_files"]]
    authors = authors.merge(pr_small, on=["repo", "pr_number"], how="left")

    ent = flow[["module_id_global", "month", "entrant_logins"]].explode("entrant_logins")
    ent = ent.rename(columns={"entrant_logins": "actor_login"}).dropna(subset=["actor_login"])
    merged = ent.merge(authors, on=["module_id_global", "month", "actor_login"], how="inner")
    agg = (
        merged.groupby(["module_id_global", "month"])
        .agg(entrant_pr_lines=("pr_lines", "mean"), entrant_pr_files=("pr_files", "mean"))
        .reset_index()
    )
    return flow.merge(agg, on=["module_id_global", "month"], how="left")


def fit(d: pd.DataFrame, y: str, cols: list[str]) -> dict:
    dd = d.dropna(subset=[y, *cols]).set_index(["module_id_global", "month_period"])
    if dd.empty or dd["repo"].nunique() < 5:
        return {}
    r_r = PanelOLS(dd[y], dd[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", clusters=dd["repo"]
    )
    r_m = PanelOLS(dd[y], dd[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    return {
        "y": y,
        "n_obs": int(r_r.nobs),
        "n_repos": int(dd["repo"].nunique()),
        "params": r_r.params.to_dict(),
        "se_repo": r_r.std_errors.to_dict(),
        "p_repo": r_r.pvalues.to_dict(),
        "p_module": r_m.pvalues.to_dict(),
    }


def print_table(r: dict, ev_cols: list[str]) -> None:
    if not r:
        print("  (推定できず)")
        return
    print(f"\n--- {r['y']} (n={r['n_obs']}, repos={r['n_repos']}) ---")
    print(f"{'k':>5} {'coef':>10} {'se(repo)':>10} {'p(repo)':>10} {'p(mod)':>10}")
    for c in ev_cols:
        if c not in r["params"]:
            continue
        k = ("-" + c[3:]) if c.startswith("k_m") else ("+" + c[3:])
        star = " *" if r["p_repo"][c] < 0.05 else ""
        print(f"{k:>5} {r['params'][c]:>+10.4f} {r['se_repo'][c]:>10.4f} "
              f"{r['p_repo'][c]:>10.4g} {r['p_module'][c]:>10.4g}{star}")
    print(f"{'-1':>5} {'0 (ref)':>10}")


def main() -> None:
    df = build_count_panel()
    months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(months)}
    adoption, never, always = detect_adoption(df)

    flow = build_flow_panel(months)
    flow = add_entrant_task_size(flow)
    flow = flow[flow["repo"].isin(set(adoption) | set(never))].copy()
    flow["month_period"] = pd.PeriodIndex(flow["month"], freq="M").to_timestamp()
    flow["month_idx"] = flow["month"].map(midx)
    flow["g"] = flow["repo"].map({r: midx[m] for r, m in adoption.items()})
    flow["event_time"] = flow["month_idx"] - flow["g"]

    flow["log_entrants"] = np.log1p(flow["n_entrants"])
    flow["log_active_humans"] = np.log1p(flow["n_active_humans"])
    flow["log_entrant_pr_lines"] = np.log1p(flow["entrant_pr_lines"])
    flow["log_entrant_pr_files"] = np.log1p(flow["entrant_pr_files"])

    ev_cols: list[str] = []
    for j in range(-LEAD_LAG, LEAD_LAG + 1):
        if j == REF:
            continue
        name = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
        k = flow["event_time"]
        val = (k <= j) if j == -LEAD_LAG else ((k >= j) if j == LEAD_LAG else (k == j))
        flow[name] = val.fillna(False).astype(float)
        ev_cols.append(name)

    out: dict = {
        "spec": {"burn_in": BURN_IN, "churn_k": CHURN_K},
        "adoption": adoption,
        "models": {},
    }
    print("=== 人材フロー(処置=リポジトリの AI レビュー導入) ===")
    for y in [
        "log_entrants",
        "churn_rate",
        "log_active_humans",
        "log_entrant_pr_lines",
        "log_entrant_pr_files",
    ]:
        r = fit(flow, y, ev_cols)
        out["models"][y] = r
        print_table(r, ev_cols)

    (OUT_DIR / "d8_talent_flow.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd8_talent_flow.json'}")


if __name__ == "__main__":
    main()

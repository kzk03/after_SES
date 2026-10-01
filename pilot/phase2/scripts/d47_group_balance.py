"""D47: 処置群と未処置群の導入前の比較(査読 M12 への対応)。

共通の導入前窓(最初の導入月 2024-10 より前: 2023-01..2024-09)で、リポジトリごとの
月平均を取り、群ごとに中央値と平均を出す。記述統計のみで検定はしない。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import OUT_DIR, build_count_panel
import d4_repo_event_study as d4

PRE_START, PRE_END = "2023-01", "2024-09"


def main() -> None:
    df = build_count_panel()
    treated, never, excluded = d4.detect_adoption_presence(df)
    assert (len(treated), len(never), len(excluded)) == (17, 14, 7)
    first_adopt = min(treated.values())
    assert first_adopt > PRE_END

    pre = df[(df["month"] >= PRE_START) & (df["month"] <= PRE_END)]
    per_repo_month = pre.groupby(["repo", "month"]).agg(
        nonauthor=("n_E_human_nonauthor", "sum"),
        human=("n_E_human", "sum"),
        agent=("n_E_agent", "sum"),
        prs=("n_E_prs", "sum"),
        modules=("n_E", lambda s: int((s > 0).sum())),
    ).reset_index()
    per_repo_month["author_share"] = 1 - per_repo_month["nonauthor"] / per_repo_month["human"].where(per_repo_month["human"] > 0)

    def growth(g: pd.DataFrame) -> float:
        a = g[g["month"] <= "2023-12"]["nonauthor"].mean()
        b = g[g["month"] >= "2024-01"]["nonauthor"].mean()
        return float(np.log(b / a)) if a > 0 and b > 0 else np.nan

    repo = per_repo_month.groupby("repo").agg(
        nonauthor=("nonauthor", "mean"), agent=("agent", "mean"), prs=("prs", "mean"),
        modules=("modules", "mean"), author_share=("author_share", "mean"), months=("month", "nunique"),
    )
    repo["growth_2023_to_2024"] = per_repo_month.groupby("repo").apply(growth)

    out = {"window": [PRE_START, PRE_END], "first_adoption": first_adopt, "groups": {}}
    for name, members in [("treated", list(treated)), ("never_treated", list(never))]:
        r = repo.loc[repo.index.intersection(members)]
        out["groups"][name] = {
            "n_repos": int(len(r)),
            **{f"{c}_median": round(float(r[c].median()), 4) for c in r.columns},
            **{f"{c}_mean": round(float(r[c].mean()), 4) for c in r.columns},
        }
        print(name, json.dumps(out["groups"][name], indent=1))
    (OUT_DIR / "d47_group_balance.json").write_text(json.dumps(out, indent=1))
    print("->", OUT_DIR / "d47_group_balance.json")


if __name__ == "__main__":
    main()

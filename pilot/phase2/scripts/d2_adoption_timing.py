"""D2: AI 参入タイミングの診断 — イベントスタディが組めるかを確認する。

D1 で「H_E の低下は dilution(AI の上乗せ)である」ことが分かったが、
固定効果回帰である以上「AI 関与度が高いモジュール」の内生性(AI が入りやすい性質の
モジュールが元々違う、レビューが大変だから AI を入れた、等)は排除できていない。

イベントスタディ/DiD に進む前に、以下を診断する。
  1. 2026-02 に見えた断層はリポジトリ横断的か(=外生的なプラットフォーム側の変更か)
  2. モジュール単位の「AI 初参入月」はばらついているか(staggered adoption が成立するか)
  3. never-treated(期間中ずっと AI が入らない)モジュールは十分あるか(対照群が取れるか)

実行: uv run python pilot/phase2/scripts/d2_adoption_timing.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import OUT_DIR, build_count_panel


def first_month_with(df: pd.DataFrame, cond_col: str) -> pd.Series:
    """モジュールごとに cond_col が True になった最初の月を返す(無ければ NaN)。"""
    hit = df[df[cond_col]]
    return hit.groupby("module_id_global")["month"].min()


def main() -> None:
    df = build_count_panel()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    # --- 1. 2026-02 の断層はリポジトリ横断的か ------------------------------------
    by_repo_month = (
        df.groupby(["repo", "month"])[["n_E_agent", "n_E_human", "n_G_agent", "n_G_human_strict"]]
        .sum()
        .reset_index()
    )
    by_repo_month["agent_review_share"] = by_repo_month["n_E_agent"] / (
        by_repo_month["n_E_agent"] + by_repo_month["n_E_human"]
    )
    pivot = by_repo_month.pivot(index="month", columns="repo", values="agent_review_share")
    pivot.to_csv(OUT_DIR / "d2_agent_review_share_by_repo.csv")

    # 「2026-02以降の平均 - 2026-01以前の平均」がプラスになるリポジトリの割合
    pre = pivot.loc[pivot.index <= "2026-01"].mean()
    post = pivot.loc[pivot.index >= "2026-02"].mean()
    jump = (post - pre).dropna()
    out["step_2026_02"] = {
        "n_repos_evaluated": int(len(jump)),
        "n_repos_increased": int((jump > 0).sum()),
        "share_increased": float((jump > 0).mean()),
        "median_jump": float(jump.median()),
        "per_repo_jump": {k: float(v) for k, v in jump.sort_values(ascending=False).items()},
    }

    # 断層が「2026-02」に集中しているか、月ごとの全体増分で確認
    agg = df.groupby("month")[["n_E_agent", "n_E_human"]].sum()
    agg["agent_share"] = agg["n_E_agent"] / (agg["n_E_agent"] + agg["n_E_human"])
    out["monthly_agent_review_share"] = agg["agent_share"].to_dict()
    out["monthly_share_diff"] = agg["agent_share"].diff().dropna().to_dict()

    # --- 2. モジュール単位の AI 初参入月 -------------------------------------------
    df = df.sort_values(["module_id_global", "month"])
    df["has_agent_E"] = df["n_E_agent"] > 0
    df["has_agent_G"] = df["n_G_agent"] > 0
    # 単発ノイズを避けた定義: AI レビューが2件以上入った月
    df["has_agent_E_strong"] = df["n_E_agent"] >= 2

    n_modules = df["module_id_global"].nunique()
    for tag, col in [
        ("agent_review_any", "has_agent_E"),
        ("agent_review_2plus", "has_agent_E_strong"),
        ("agent_generation_any", "has_agent_G"),
    ]:
        g = first_month_with(df, col)
        out[f"adoption_{tag}"] = {
            "n_treated_modules": int(len(g)),
            "n_never_treated": int(n_modules - len(g)),
            "never_treated_share": float((n_modules - len(g)) / n_modules),
            "cohort_sizes": {k: int(v) for k, v in g.value_counts().sort_index().items()},
        }

    # --- 3. 対照群の質: never-treated モジュールの規模は treated と比べてどうか ------
    g_any = first_month_with(df, "has_agent_E")
    mod_size = df.groupby("module_id_global").agg(
        n_months=("month", "nunique"),
        total_E=("n_E", "sum"),
        total_G=("n_G", "sum"),
    )
    mod_size["treated"] = mod_size.index.isin(g_any.index)
    out["treated_vs_never"] = {
        "treated": {
            "n": int(mod_size["treated"].sum()),
            "median_total_E": float(mod_size.loc[mod_size["treated"], "total_E"].median()),
            "median_total_G": float(mod_size.loc[mod_size["treated"], "total_G"].median()),
            "median_n_months": float(mod_size.loc[mod_size["treated"], "n_months"].median()),
        },
        "never_treated": {
            "n": int((~mod_size["treated"]).sum()),
            "median_total_E": float(mod_size.loc[~mod_size["treated"], "total_E"].median()),
            "median_total_G": float(mod_size.loc[~mod_size["treated"], "total_G"].median()),
            "median_n_months": float(mod_size.loc[~mod_size["treated"], "n_months"].median()),
        },
    }

    (OUT_DIR / "d2_adoption_timing.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    # --- 表示 ---------------------------------------------------------------------
    print("=== 1. 2026-02 の断層はリポジトリ横断的か ===")
    s = out["step_2026_02"]
    print(f"AI レビュー比率が 2026-02 以降に上がったリポジトリ: {s['n_repos_increased']}/{s['n_repos_evaluated']} "
          f"({s['share_increased']:.0%})、中央値の増分 {s['median_jump']:+.4f}")
    print("\n月次 AI レビュー比率(全体):")
    for m, v in out["monthly_agent_review_share"].items():
        d = out["monthly_share_diff"].get(m)
        print(f"  {m}: {v:.4f}" + (f"  (前月差 {d:+.4f})" if d is not None else ""))

    print("\n=== 2. モジュール単位の AI 初参入月(staggered adoption) ===")
    for tag in ["agent_review_any", "agent_review_2plus", "agent_generation_any"]:
        a = out[f"adoption_{tag}"]
        print(f"\n[{tag}] treated={a['n_treated_modules']}, never-treated={a['n_never_treated']} "
              f"({a['never_treated_share']:.0%})")
        print("  コホート(初参入月別モジュール数):")
        for m, n in a["cohort_sizes"].items():
            print(f"    {m}: {n}")

    print("\n=== 3. treated vs never-treated のモジュール規模 ===")
    print(json.dumps(out["treated_vs_never"], ensure_ascii=False, indent=2))
    print(f"\n-> {OUT_DIR / 'd2_adoption_timing.json'}")


if __name__ == "__main__":
    main()

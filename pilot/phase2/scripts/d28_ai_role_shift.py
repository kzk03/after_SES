"""D28: ★AI 自身の役割は変遷しているのか。

ここまで人間側の振る舞いを5次元(量・役割配分・審査の厳しさ・負荷分配・タイミング)で見て、
いずれも動いていないことを確認した。しかし **AI 側**の振る舞いは見ていない。

これは論文の主張に直結する。
「AI の判断率 0.27%」が **現時点のスナップショット**なのか **構造的な境界**なのかで、含意が変わる。

  - 判断率が上がりつつある → 「回路が開き始めている」。主張に時間的な限定が必要
  - 横ばい               → 「境界は動いていない」。構造的な主張が強まる

5つの軸で AI 側の変遷を見る。
  1. 判断率(APPROVED + CHANGES_REQUESTED の割合)の月次推移
  2. コメントの分量の推移(初期は長く、最近は短くなったか)
  3. 指摘が対処される率(outdated 率)の推移 = ツールの改善
  4. エージェントの世代交代(誰が主役か)
  5. 見ている対象の変化(テストファイル / 設定ファイル / ドキュメントの比率)

特に注目: **claude だけ判断率 11.9%** と他(0〜0.1%)より高い。
このエージェントのシェアが伸びているなら、境界が動く兆しかもしれない。

実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d28_ai_role_shift.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import OUT_DIR


def trend_test(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> dict:
    """月インデックスに対する線形トレンドを検定する(件数で重み付け可)。"""
    X = sm.add_constant(x)
    if w is not None:
        m = sm.WLS(y, X, weights=w).fit()
    else:
        m = sm.OLS(y, X).fit()
    return {"slope": float(m.params[1]), "p": float(m.pvalues[1]), "n": int(m.nobs)}


def main() -> None:
    out: dict = {}

    # =====================================================================
    # 1. 判断率の月次推移(d5_pr_level から)
    # =====================================================================
    pr = pd.read_parquet(OUT_DIR / "d5_pr_level.parquet")
    m = pr.groupby("month").agg(
        ai_appr=("n_ai_approved", "sum"),
        ai_cr=("n_ai_changes_requested", "sum"),
        ai_com=("n_ai_commented", "sum"),
        hu_appr=("n_human_approved", "sum"),
        hu_cr=("n_human_changes_requested", "sum"),
        hu_com=("n_human_commented", "sum"),
    ).reset_index()
    m["ai_total"] = m["ai_appr"] + m["ai_cr"] + m["ai_com"]
    m["hu_total"] = m["hu_appr"] + m["hu_cr"] + m["hu_com"]
    m = m[m["ai_total"] >= 50].reset_index(drop=True)
    m["ai_verdict_rate"] = (m["ai_appr"] + m["ai_cr"]) / m["ai_total"]
    m["hu_verdict_rate"] = (m["hu_appr"] + m["hu_cr"]) / m["hu_total"]

    print("=" * 74)
    print("【1】AI の判断率(APPROVED + CHANGES_REQUESTED の割合)は上がっているか")
    print("=" * 74)
    print(f"  {'month':<10}{'AIレビュー数':>12}{'AI判断率':>12}{'(参考)人間判断率':>18}")
    for _, r in m.iloc[::3].iterrows():
        print(f"  {r['month']:<10}{int(r['ai_total']):>12,}{r['ai_verdict_rate']:>12.3%}"
              f"{r['hu_verdict_rate']:>18.1%}")
    t = trend_test(np.arange(len(m)), m["ai_verdict_rate"].values, m["ai_total"].values)
    out["verdict_rate_trend"] = t
    print(f"\n  トレンド(件数重み付け): 月あたり {t['slope']:+.5f} (p={t['p']:.4g}, n={t['n']})")
    print(f"  期間全体: {m['ai_verdict_rate'].iloc[0]:.3%} -> {m['ai_verdict_rate'].iloc[-1]:.3%}")
    if t["p"] < 0.05 and t["slope"] > 0:
        print("  → ★上昇傾向。判断の回路が開き始めている可能性")
    elif t["p"] < 0.05:
        print("  → 低下傾向")
    else:
        print("  → **有意なトレンドなし。境界は動いていない**")

    # =====================================================================
    # 2〜5. コメント単位の推移(d26_triage から)
    # =====================================================================
    tri = pd.read_parquet(OUT_DIR / "d26_triage.parquet")
    tri["month"] = tri["created_at"].str[:7]
    g = tri.groupby("month").agg(
        n=("outdated", "size"),
        len_visible=("len_visible", "mean"),
        outdated=("outdated", "mean"),
        has_suggestion=("has_suggestion", "mean"),
        has_severity=("has_severity", "mean"),
        is_test=("is_test_file", "mean"),
        is_config=("is_config", "mean"),
        is_doc=("is_doc", "mean"),
    ).reset_index()
    g = g[g["n"] >= 100].reset_index(drop=True)

    print("\n" + "=" * 74)
    print("【2-3】コメントの分量と、指摘が対処される率は変わったか")
    print("=" * 74)
    print(f"  {'month':<10}{'コメント数':>11}{'平均文字数':>11}{'対処率':>9}{'suggestion率':>14}")
    for _, r in g.iloc[::3].iterrows():
        print(f"  {r['month']:<10}{int(r['n']):>11,}{r['len_visible']:>11.0f}"
              f"{r['outdated']:>9.1%}{r['has_suggestion']:>14.1%}")
    x = np.arange(len(g))
    for col, lab in [("len_visible", "平均文字数"), ("outdated", "対処率"),
                     ("has_suggestion", "suggestion 形式の割合")]:
        t = trend_test(x, g[col].values, g["n"].values)
        out[f"trend_{col}"] = t
        arrow = "↑" if (t["p"] < 0.05 and t["slope"] > 0) else ("↓" if t["p"] < 0.05 else "→")
        print(f"\n  {lab:<22}{g[col].iloc[0]:>10.3f} -> {g[col].iloc[-1]:>8.3f}  "
              f"月あたり {t['slope']:+.5f} (p={t['p']:.4g}) {arrow}")

    # =====================================================================
    # 4. エージェントの世代交代
    # =====================================================================
    print("\n" + "=" * 74)
    print("【4】エージェントの世代交代 — 誰が主役か")
    print("=" * 74)
    sh = tri.groupby(["month", "agent"]).size().rename("n").reset_index()
    tot = sh.groupby("month")["n"].transform("sum")
    sh["share"] = sh["n"] / tot
    piv = sh.pivot(index="month", columns="agent", values="share").fillna(0)
    piv = piv[piv.index.isin(g["month"])]
    cols = piv.sum().sort_values(ascending=False).index[:5]
    print(f"  {'month':<10}" + "".join(f"{c[:22]:>24}" for c in cols))
    for mo in piv.index[::3]:
        print(f"  {mo:<10}" + "".join(f"{piv.loc[mo, c]:>24.1%}" for c in cols))
    out["agent_share"] = piv[cols].to_dict()

    # claude(判断を出す唯一のエージェント)のシェアは伸びているか
    if "claude" in piv.columns:
        t = trend_test(np.arange(len(piv)), piv["claude"].values)
        out["claude_share_trend"] = t
        print(f"\n  ★判断を出す唯一のエージェント claude のシェア: "
              f"{piv['claude'].iloc[0]:.1%} -> {piv['claude'].iloc[-1]:.1%}  "
              f"(トレンド {t['slope']:+.5f}/月, p={t['p']:.4g})")

    # =====================================================================
    # 5. 見ている対象の変化
    # =====================================================================
    print("\n" + "=" * 74)
    print("【5】AI が指摘する対象は変わったか(ファイル種別)")
    print("=" * 74)
    print(f"  {'month':<10}{'テスト':>10}{'設定':>10}{'ドキュメント':>14}")
    for _, r in g.iloc[::3].iterrows():
        print(f"  {r['month']:<10}{r['is_test']:>10.1%}{r['is_config']:>10.1%}{r['is_doc']:>14.1%}")
    for col, lab in [("is_test", "テストファイル比率"), ("is_config", "設定ファイル比率"),
                     ("is_doc", "ドキュメント比率")]:
        t = trend_test(x, g[col].values, g["n"].values)
        out[f"trend_{col}"] = t
        arrow = "↑" if (t["p"] < 0.05 and t["slope"] > 0) else ("↓" if t["p"] < 0.05 else "→")
        print(f"  {lab:<22}月あたり {t['slope']:+.5f} (p={t['p']:.4g}) {arrow}")

    (OUT_DIR / "d28_ai_role_shift.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'd28_ai_role_shift.json'}")


if __name__ == "__main__":
    main()

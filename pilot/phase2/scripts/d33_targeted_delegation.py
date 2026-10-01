"""D33: 委譲を絞る方策に価値はあるか — AI 導入効果の異質性と方策評価。

D4 は「AI 導入は人間のレビューを減らさない」を平均として示した。
方策の問いはその先にある。**平均がゼロでも、部分集合では負かもしれない。**
負の部分集合があるなら、そこに委譲を絞る方策には価値がある。無ければ無い。

これは Athey & Wager 的な方策学習の枠組みで、
  1. 異質効果 CATE(g) を識別のある変動(リポジトリ単位の AI 導入)で推定する
  2. CATE が負の群だけに委譲する方策の価値を、一律委譲と比べる
という二段構えになる。

識別は D4 と同じ。`RL_REFRAME.md` §1 のとおり、
**割り当ての margin には外生変動が無いが、委譲の margin にはある**。

層別の軸:
  (a) モジュールの種別(テスト / ドキュメント / 設定 / コード)
  (b) 導入前のレビュー密度(四分位)
  (c) モジュールの規模

⚠ 交互作用ではなく部分集合ごとの DiD なので、staggered adoption の
   TWFE バイアス(Goodman-Bacon)は残る。D6 の stacked DiD と併せて読む。

出力: outputs_full/d33_targeted_delegation.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d33_targeted_delegation.py
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import OUT_DIR, build_count_panel, fit_panel

DENSITY_WINDOW_END = "2025-01"   # 密度を測る共通の暦窓の終わり

TEST_PAT = re.compile(r"(^|/)(tests?|spec|e2e|__tests__|testing)(/|$)|_test\.|\.test\.", re.I)
DOC_PAT = re.compile(r"(^|/)(docs?|documentation|website|examples?)(/|$)|\.md$|\.rst$", re.I)
CFG_PAT = re.compile(r"(^|/)(\.github|ci|config|build|scripts|tools|vendor|deps)(/|$)"
                     r"|\.ya?ml$|\.json$|\.toml$|\.cfg$|Dockerfile", re.I)


def month_to_idx(s: str) -> int:
    y, m = str(s).split("-")[:2]
    return int(y) * 12 + int(m)


def module_kind(module_id: str) -> str:
    """module_id は "L677:/pkg/networkdriver" のような形。パス部分で種別を判定する。"""
    path = module_id.split(":", 1)[-1]
    if TEST_PAT.search(path):
        return "テスト"
    if DOC_PAT.search(path):
        return "ドキュメント"
    if CFG_PAT.search(path):
        return "設定・ビルド"
    return "コード"


def main() -> None:
    adoption = json.loads((OUT_DIR / "d4_repo_event_study.json").read_text())["adoption"]
    # ★対照は never-treated のみ。判定不能で除外したリポジトリを対照に混ぜない(2026-09-27 数値表の指摘)
    _d4 = json.loads((OUT_DIR / "d4_repo_event_study.json").read_text())
    KEEP_REPOS = set(adoption) | set(_d4["never_treated"])

    df = build_count_panel()
    df = df[df["repo"].isin(KEEP_REPOS)].copy()
    df["t"] = df["month"].map(month_to_idx)
    df["adopt_t"] = df["repo"].map(lambda r: month_to_idx(adoption[r]) if r in adoption else np.nan)
    df["treated_repo"] = df["adopt_t"].notna()
    df["post"] = (df["treated_repo"] & (df["t"] >= df["adopt_t"])).astype(float)

    df["log_n_E_human"] = np.log1p(df["n_E_human"])
    df["log_n_E_agent"] = np.log1p(df["n_E_agent"])
    df["e_human_per_pr"] = np.where(df["n_E_prs"] > 0, df["n_E_human"] / df["n_E_prs"], np.nan)

    df["kind"] = df["module_id"].map(module_kind)

    # 導入前のレビュー密度で四分位に分ける。
    # ★全モジュールで同じ暦窓(2025-01 より前)を使う。当初は処置群を導入前、対照群を全期間で
    #   計算しており、処置群の「疎」だけが平均回帰で上がる仕組みになっていた(偽処置で +0.734)。
    #   2026-09-27 の査読(R1)で発覚して修正。
    pre = df[df["month"] < DENSITY_WINDOW_END]
    dens = pre.groupby("module_id_global")["n_E_human"].mean().rename("pre_density")
    df = df.merge(dens, on="module_id_global", how="left")
    df["dens_q"] = pd.qcut(df["pre_density"], 4, labels=["Q1(疎)", "Q2", "Q3", "Q4(密)"],
                           duplicates="drop")

    out: dict = {"n_panel": int(len(df)), "adoption": adoption, "groups": {}}

    print(f"パネル {len(df):,} 行 / モジュール {df['module_id_global'].nunique():,}")
    print(f"処置リポジトリ {df[df['treated_repo']]['repo'].nunique()} / "
          f"対照 {df[~df['treated_repo']]['repo'].nunique()}\n")

    print("=== モジュール種別の構成 ===")
    print(df.groupby("kind")["module_id_global"].nunique().to_string())

    def run(label: str, sub: pd.DataFrame) -> None:
        if sub["post"].sum() < 200 or sub["repo"].nunique() < 5:
            print(f"  {label:16s} 観測不足(post={sub['post'].sum():.0f}, "
                  f"repos={sub['repo'].nunique()})")
            return
        res = {}
        for y in ["log_n_E_human", "e_human_per_pr", "log_n_E_agent"]:
            try:
                res[y] = fit_panel(sub, y, ["post"])
            except Exception as e:
                res[y] = {"error": str(e)}
        out["groups"][label] = res
        h = res.get("log_n_E_human", {})
        a = res.get("log_n_E_agent", {})
        if "params" in h:
            star = " *" if h["p_repo"]["post"] < 0.05 else ""
            first = (f"人間 {h['params']['post']:+.4f} (p={h['p_repo']['post']:.3f}){star}")
            second = (f"AI {a['params']['post']:+.4f}" if "params" in a else "")
            print(f"  {label:16s} n={h['n_obs']:>7,} repos={h['n_repos']:>2}  {first:34s} {second}")

    print("\n=== (a) モジュール種別ごとの AI 導入効果 ===")
    print("  被説明変数 = log(1+人間のレビュー件数)。負なら代替が起きている")
    for k in ["コード", "テスト", "設定・ビルド", "ドキュメント"]:
        run(k, df[df["kind"] == k])

    print("\n=== (b) 導入前のレビュー密度ごと ===")
    for q in ["Q1(疎)", "Q2", "Q3", "Q4(密)"]:
        run(q, df[df["dens_q"] == q])

    print("\n=== (c) 全体(参考) ===")
    run("全体", df)

    # --- 異質性が有意かを形式的に検定する(部分集合ごとの推定は検定になっていない)---
    print("\n=== (d) 異質性の検定: post × 導入前レビュー密度 ===")
    d2 = df.dropna(subset=["pre_density"]).copy()
    d2["ld_c"] = np.log1p(d2["pre_density"]) - np.log1p(d2["pre_density"]).mean()
    d2["post_x_ld"] = d2["post"] * d2["ld_c"]
    r_cont = fit_panel(d2, "log_n_E_human", ["post", "post_x_ld"])
    for k in ["post", "post_x_ld"]:
        star = " *" if r_cont["p_repo"][k] < 0.05 else ""
        print(f"  {k:12s} {r_cont['params'][k]:+.4f} "
              f"(se={r_cont['se_repo'][k]:.4f}, p={r_cont['p_repo'][k]:.4g}){star}")

    d2["q"] = pd.qcut(d2["pre_density"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    for q in ["Q2", "Q3", "Q4"]:
        d2[f"post_{q}"] = d2["post"] * (d2["q"] == q).astype(float)
    r_q = fit_panel(d2, "log_n_E_human", ["post", "post_Q2", "post_Q3", "post_Q4"])
    print("  四分位ダミー(基準 Q1=疎):")
    for k in ["post", "post_Q2", "post_Q3", "post_Q4"]:
        star = " *" if r_q["p_repo"][k] < 0.05 else ""
        print(f"    {k:10s} {r_q['params'][k]:+.4f} (p={r_q['p_repo'][k]:.4g}){star}")
    out["heterogeneity"] = {"continuous": r_cont, "quartile": r_q}

    # --- 方策の価値: 負の群だけに委譲する方策 vs 一律 ---
    print("\n=== 方策評価: CATE が負の群に絞る価値はあるか ===")
    neg = [(g, r["log_n_E_human"]["params"]["post"], r["log_n_E_human"]["p_repo"]["post"])
           for g, r in out["groups"].items()
           if g not in ("全体",) and "params" in r.get("log_n_E_human", {})]
    neg_sig = [(g, c, p) for g, c, p in neg if c < 0 and p < 0.05]
    print(f"  推定した群: {len(neg)}")
    print(f"  係数が負かつ有意な群: {len(neg_sig)}")
    if neg_sig:
        for g, c, p in neg_sig:
            print(f"    {g}: {c:+.4f} (p={p:.4f})")
        print("  → この群に委譲を絞る方策には価値がありうる。")
    else:
        print("  → **負で有意な群は無い。委譲を絞っても人間の作業は減らない。**")
        print("     方策の改善余地は、少なくともこの層別軸の上には存在しない。")
    out["policy"] = {"n_groups": len(neg), "n_negative_significant": len(neg_sig),
                     "negative_groups": [{"group": g, "coef": c, "p": p} for g, c, p in neg_sig]}

    (OUT_DIR / "d33_targeted_delegation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n-> {OUT_DIR / 'd33_targeted_delegation.json'}")


if __name__ == "__main__":
    main()

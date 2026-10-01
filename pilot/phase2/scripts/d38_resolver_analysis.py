"""D38: 誰がスレッドを閉じたのか — D36 の結論を直接検定する。

D36 は「同じ PR の同じ行でも、AI が立てたスレッドの方が多く解決済みにされる」を示し、
そこから「人間が機械の指摘を早く切り捨てている」と読んだ。**これは誤りだった。**

D37 で `resolvedBy` を収集したところ、解決済み AI スレッドの4分の1は
エージェントのアカウントが閉じていた(人間のスレッドでは 0.04%)。
「人間が閉じた」に限定すると効果は消える。

本スクリプトはその検定を再現可能な形にまとめる。

  1. 誰が誰のスレッドを閉じたか(クロス表)
  2. 自己解決(立てた本人が閉じた)の割合
  3. エージェント別の「人間が閉じた割合」
  4. ★同一箇所の対で「解決された」対「人間が解決した」を比べる
  5. 全標本(対に限らない)での PR 固定効果つき回帰

標本条件は D36 と揃える(`line` があるスレッド = その行のコードがまだ変わっていない)。

出力: outputs_full/d38_resolver_analysis.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d38_resolver_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name

TOL = 3


def _norm(login: str) -> str:
    return login.lower().replace("[bot]", "")


def load(require_line: bool = True) -> pd.DataFrame:
    rows = []
    for repo in load_repos():
        p = RAW_DIR / repo_dir_name(repo) / "thread_resolvers.jsonl"
        ap = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not (p.exists() and ap.exists()):
            continue
        a = pd.read_parquet(ap)
        cat = dict(zip(a["actor_login"], a["category"]))
        with p.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for th in d.get("threads") or []:
                    st, ln = th.get("starter"), th.get("line")
                    if st is None:
                        continue
                    if require_line and ln is None:
                        continue
                    sc = cat.get(st, "unknown")
                    if sc not in ("human", "agent"):
                        continue
                    rb = th.get("resolvedBy")
                    rows.append({
                        "repo": repo, "pr_number": d["number"],
                        "path": th.get("path") or "", "line": int(ln) if ln is not None else -1,
                        "starter": st, "starter_cat": sc, "is_ai": float(sc == "agent"),
                        "resolved": float(bool(th.get("isResolved"))),
                        "outdated": float(bool(th.get("isOutdated"))),
                        "resolver": rb,
                        "resolver_cat": cat.get(rb, "unknown") if rb else None,
                        # ★GraphQL は立てた側を "cursor"、閉じた側を "cursor[bot]" で返す。
                        #   正規化せずに比べると自己解決が1件も一致しない(2026-09-27 査読で発覚)
                        "self_resolved": float(rb is not None and _norm(rb) == _norm(st)),
                    })
    df = pd.DataFrame(rows)
    df["human_resolved"] = ((df["resolved"] == 1) & (df["resolver_cat"] == "human")).astype(float)
    df["pid"] = df["repo"] + "#" + df["pr_number"].astype(str)
    return df


def add_pair_flags(df: pd.DataFrame) -> pd.DataFrame:
    """同じ PR の同じ path で、反対カテゴリのスレッドが line±TOL に居るか。"""
    flags = np.zeros(len(df), dtype=float)
    pos_of = {ix: i for i, ix in enumerate(df.index)}
    for _, g in df.groupby(["repo", "pr_number", "path"], sort=False):
        ai = g.loc[g["is_ai"] == 1, "line"].to_numpy()
        hu = g.loc[g["is_ai"] == 0, "line"].to_numpy()
        if len(ai) == 0 or len(hu) == 0:
            continue
        for ix, ln, isai in zip(g.index, g["line"], g["is_ai"]):
            other = hu if isai == 1 else ai
            if np.any(np.abs(other - ln) <= TOL):
                flags[pos_of[ix]] = 1.0
    df = df.copy()
    df["has_counterpart"] = flags
    return df


def within(df: pd.DataFrame, cols: list[str], by: str = "pid") -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        d[c + "_dm"] = d[c] - d.groupby(by)[c].transform("mean")
    return d


def main() -> None:
    df = load(require_line=True)
    full = load(require_line=False)
    out: dict = {"n_threads_live_line": int(len(df)), "n_threads_all": int(len(full))}
    print(f"=== スレッド {len(df):,} 件(line あり)/ {len(full):,} 件(全て)"
          f" / PR {df['pid'].nunique():,} / リポジトリ {df['repo'].nunique()} ===")
    print(f"  AI {int(df.is_ai.sum()):,} / 人間 {int((1 - df.is_ai).sum()):,}")

    print("\n=== 1. 誰が誰のスレッドを閉じたか(解決済みのみ、全標本)===")
    res = full[full["resolved"] == 1]
    tab = pd.crosstab(res["starter_cat"], res["resolver_cat"].fillna("(不明)"), normalize="index")
    cnt = pd.crosstab(res["starter_cat"], res["resolver_cat"].fillna("(不明)"))
    print(tab.to_string(float_format=lambda x: f"{x:9.4f}"))
    out["who_resolves"] = {k: {c: float(v) for c, v in r.items()} for k, r in tab.iterrows()}
    out["who_resolves_n"] = {k: {c: int(v) for c, v in r.items()} for k, r in cnt.iterrows()}

    ai_res = res[res["is_ai"] == 1]
    print(f"\n=== 2. 自己解決(立てた本人が閉じた)===")
    print(f"  AI スレッド {len(ai_res):,} 件中 {int(ai_res.self_resolved.sum()):,} 件 "
          f"({ai_res.self_resolved.mean():.4%})")
    out["self_resolved_share"] = float(ai_res["self_resolved"].mean())

    print("\n=== 3. エージェント別の「人間が閉じた割合」===")
    g = ai_res.groupby("starter").agg(
        n=("self_resolved", "size"),
        人間が解決=("resolver_cat", lambda s: (s == "human").mean()),
        自己解決=("self_resolved", "mean")).sort_values("n", ascending=False)
    print(g.head(10).to_string(float_format=lambda x: f"{x:9.4f}"))
    out["by_agent"] = {k: {"n": int(v["n"]), "human_resolved": float(v["人間が解決"])}
                       for k, v in g.iterrows()}

    print("\n=== 4. ★同一箇所の対 ===")
    dfp = add_pair_flags(df)
    pair = dfp[dfp["has_counterpart"] == 1].copy()
    print(f"  対 {len(pair):,} 件(AI {int(pair.is_ai.sum()):,} / "
          f"人間 {int((1 - pair.is_ai).sum()):,}) / PR {pair['pid'].nunique():,}")
    gp = pair.groupby("is_ai")[["resolved", "human_resolved"]].mean().rename(
        index={0.0: "人間", 1.0: "AI"})
    print(gp.to_string(float_format=lambda x: f"{x:9.4f}"))
    pw = within(pair, ["resolved", "human_resolved", "is_ai"])
    out["pairs"] = {}
    for y, lab in [("resolved", "解決された(誰でも)"), ("human_resolved", "★人間が解決した")]:
        m = smf.ols(f"{y}_dm ~ is_ai_dm - 1", data=pw).fit(
            cov_type="cluster", cov_kwds={"groups": pw["repo"]})
        star = " *" if m.pvalues["is_ai_dm"] < 0.05 else ""
        print(f"    {lab:20s} is_ai {m.params['is_ai_dm']:+.4f} "
              f"(se={m.bse['is_ai_dm']:.4f}, p={m.pvalues['is_ai_dm']:.4g}) n={int(m.nobs):,}{star}")
        out["pairs"][lab] = {"coef": float(m.params["is_ai_dm"]),
                             "se": float(m.bse["is_ai_dm"]),
                             "p": float(m.pvalues["is_ai_dm"]), "n": int(m.nobs)}

    print("\n=== 5. 全標本(対に限らない)の PR 固定効果つき回帰 ===")
    mixed = df.groupby("pid")["is_ai"].transform("nunique") > 1
    dm = within(df[mixed], ["resolved", "human_resolved", "is_ai"])
    print(f"  両方が居る PR: {dm['pid'].nunique():,} / スレッド {len(dm):,}")
    out["full_sample"] = {}
    for y, lab in [("resolved", "解決された(誰でも)"), ("human_resolved", "★人間が解決した")]:
        m = smf.ols(f"{y}_dm ~ is_ai_dm - 1", data=dm).fit(
            cov_type="cluster", cov_kwds={"groups": dm["repo"]})
        star = " *" if m.pvalues["is_ai_dm"] < 0.05 else ""
        print(f"    {lab:20s} is_ai {m.params['is_ai_dm']:+.4f} "
              f"(se={m.bse['is_ai_dm']:.4f}, p={m.pvalues['is_ai_dm']:.4g}) n={int(m.nobs):,}{star}")
        out["full_sample"][lab] = {"coef": float(m.params["is_ai_dm"]),
                                   "se": float(m.bse["is_ai_dm"]),
                                   "p": float(m.pvalues["is_ai_dm"]), "n": int(m.nobs)}

    # 同値性: 「人間が解決した」の差が δ 以上あるとは言えるか
    base = df[df["is_ai"] == 0]["human_resolved"].mean()
    c = out["full_sample"]["★人間が解決した"]["coef"]
    se = out["full_sample"]["★人間が解決した"]["se"]
    print(f"\n  人間スレッドの基準率 {base:.4f} / MDE(相対) {(1.96+1.64)*se/base:.1%}")
    from scipy.stats import norm
    for d_rel in [0.05, 0.10, 0.20]:
        delta = d_rel * base
        p_hi = 1 - norm.cdf((delta - c) / se)      # H0: 差 >= +delta
        p_lo = norm.cdf((-delta - c) / se)         # H0: 差 <= -delta
        print(f"    |差| >= {d_rel:.0%} ({delta:+.4f}): 上側 p={p_hi:.3g} / 下側 p={p_lo:.3g}")
        out.setdefault("equivalence", {})[f"{d_rel:.0%}"] = {"p_upper": float(p_hi),
                                                             "p_lower": float(p_lo)}

    (OUT_DIR / "d38_resolver_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n-> {OUT_DIR / 'd38_resolver_analysis.json'}")


if __name__ == "__main__":
    main()

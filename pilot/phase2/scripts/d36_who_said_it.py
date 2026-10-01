"""D36: 同じ指摘でも、誰が言ったかで扱いが変わるのか。

本研究の中心的な主張は「AI は発言するが判断しない」である。だが
**なぜ AI の発言が人間の作業を減らさないのか**は説明できていない。候補は2つある。

  (a) 内容が劣るから効かない
  (b) 発話者が機械だから効かない

この2つを分離できれば、主張は「AI は判断に参加していない」から
「同じ指摘でも、人間が言えば動き、機械が言えば動かない」に変わる。

> ❌ **2026-09-24 撤回。本スクリプトの結論は誤りだった。**
> 本スクリプトは `isResolved` を被説明変数にして「AI のスレッドの方が多く閉じられる」
> (同一箇所の対で +0.111, p=5.4e-06)を示し、そこから
> 「人間が機械の指摘を早く切り捨てている」と読んだ。
>
> **`resolvedBy` を収集したら覆った(D37/D38)。**
> 解決済み AI スレッドの **26.1% はエージェントのアカウントが閉じていた**
> (人間のスレッドでは 0.04%)。しかも立てた本人が閉じた例は 0 件で、別のエージェントである。
> 「人間が閉じた」に限定すると効果は **+0.050 (p=0.207)** で消える。
>
> **人間は AI のスレッドと人間のスレッドを区別できない率で閉じている。**
> 本スクリプトは「素朴な推定値」を再現する目的で残す。
> 結論は `d38_resolver_analysis.py` を見ること。
>
> 教訓: 対抗説明(エージェント間のばらつき・リポジトリ別・コメント長)を
> **間接的に3つ潰しただけで満足した**。直接データを取れば済む話だった。

`thread_status.jsonl` はレビュースレッド単位で
  starter(発話者の login)/ path / line / isOutdated / isResolved
を持つ。

★**標本の性質を先に押さえる。** GitHub はスレッドが outdated になると `line` を null にする。
実測でも `line` あり → outdated 7.8%、`line` なし → outdated 99.1% で、ほぼ表裏である。
**同じ箇所を突き合わせるには `line` が要るので、本分析の標本は
「その行のコードがまだ変わっていないスレッド」に限られる。**

これは欠点ではなく、この問いにとって**正しい条件付け**である。
「コードが変わっていないのに閉じられた」= 対処せずに片付けた、を直接見られる。
ただし同じ理由で、**この標本での isOutdated の比較は構造的に無意味**なので行わない。
コードが変わったかを含む全標本での比較は D23 にある
(PR 固定効果で resolved +0.133, outdated −0.101)。

設計:
  A. スレッド単位の回帰(被説明変数 = isResolved)。PR 固定効果で「同じ PR の中での」
     比較にし、内容の観測可能な特徴(本文長・suggestion・コードブロック・ファイル種別)を統制する。
  B. ★同一箇所の対。同じ PR の同じ path・line±3 に AI と人間の両方がスレッドを立てた場合、
     **「どこを指摘したか」の交絡が消える**。コードの状態も同じである。
     それでも扱いが違うなら、差は**発話者が誰か**に帰せられる。

⚠ 割り当てはランダムでない。AI は広く撒き、人間は重要と判断した所に書く。
   観測できない内容の質は残る。**上界・下界として読む。**

出力: outputs_full/d36_who_said_it.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d36_who_said_it.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name

TOL = 3          # 「同じ箇所」とみなす行のずれ(D21 と揃える)
TEST_PAT = re.compile(r"(^|/)(tests?|spec|e2e|__tests__)(/|$)|_test\.|\.test\.", re.I)
DOC_PAT = re.compile(r"(^|/)(docs?|website|examples?)(/|$)|\.md$|\.rst$", re.I)
CFG_PAT = re.compile(r"(^|/)(\.github|ci|config|build|scripts)(/|$)|\.ya?ml$|\.json$|\.toml$", re.I)


def load_threads() -> pd.DataFrame:
    """スレッドに発話者カテゴリと、同じ PR のコメント本文を突き合わせる。"""
    rows = []
    for repo in load_repos():
        tpath = RAW_DIR / repo_dir_name(repo) / "thread_status.jsonl"
        bpath = RAW_DIR / repo_dir_name(repo) / "review_bodies.jsonl"
        apath = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not (tpath.exists() and apath.exists()):
            continue
        a = pd.read_parquet(apath)
        cat_of = dict(zip(a["actor_login"], a["category"]))

        # コメント本文を (pr, path, line) で引けるようにする
        body_at: dict[tuple[int, str, int], str] = {}
        if bpath.exists():
            with bpath.open() as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    num = d.get("number")
                    for rv in d.get("reviews") or []:
                        for c in rv.get("comments") or []:
                            ln = c.get("line") or c.get("originalLine")
                            if ln is None:
                                continue
                            body_at[(num, c.get("path") or "", int(ln))] = c.get("body") or ""

        with tpath.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                num = d.get("number")
                for th in d.get("threads") or []:
                    st = th.get("starter")
                    ln = th.get("line")
                    if st is None or ln is None:
                        continue
                    cat = cat_of.get(st, "unknown")
                    if cat not in ("human", "agent"):
                        continue          # legacy_bot / unknown は対象外
                    path = th.get("path") or ""
                    body = body_at.get((num, path, int(ln)), "")
                    rows.append({
                        "repo": repo, "pr_number": num, "path": path, "line": int(ln),
                        "starter": st, "is_ai": float(cat == "agent"),
                        "outdated": float(bool(th.get("isOutdated"))),
                        "resolved": float(bool(th.get("isResolved"))),
                        "created_at": th.get("createdAt"),
                        "body_len": len(body),
                        "has_suggestion": float("```suggestion" in body),
                        "has_code_block": float("```" in body),
                        "is_test": float(bool(TEST_PAT.search(path))),
                        "is_doc": float(bool(DOC_PAT.search(path))),
                        "is_config": float(bool(CFG_PAT.search(path))),
                        "path_depth": path.count("/"),
                    })
    return pd.DataFrame(rows)


def add_pair_flags(df: pd.DataFrame) -> pd.DataFrame:
    """同じ PR の同じ path で、反対カテゴリのスレッドが line±TOL に居るかを付ける。"""
    flags = np.zeros(len(df), dtype=float)
    idx = df.index.to_numpy()
    for (_, _, _), g in df.groupby(["repo", "pr_number", "path"], sort=False):
        ai_lines = g.loc[g["is_ai"] == 1, "line"].to_numpy()
        hu_lines = g.loc[g["is_ai"] == 0, "line"].to_numpy()
        if len(ai_lines) == 0 or len(hu_lines) == 0:
            continue
        for pos, (i, ln, isai) in enumerate(zip(g.index, g["line"], g["is_ai"])):
            other = hu_lines if isai == 1 else ai_lines
            if np.any(np.abs(other - ln) <= TOL):
                flags[np.searchsorted(idx, i)] = 1.0
    df = df.copy()
    df["has_counterpart"] = flags
    return df


def main() -> None:
    df = load_threads()
    print(f"=== スレッド {len(df):,} 件 / PR {df.groupby(['repo','pr_number']).ngroups:,} "
          f"/ リポジトリ {df['repo'].nunique()} ===")
    print(f"  人間が立てた {int((df.is_ai==0).sum()):,} / AI が立てた {int((df.is_ai==1).sum()):,}")
    print(f"  本文が突き合わせられた割合: {(df.body_len>0).mean():.1%}")

    print("\n=== 記述: 発話者別のスレッドの帰趨 ===")
    g = df.groupby("is_ai").agg(
        n=("outdated", "size"), outdated=("outdated", "mean"),
        resolved=("resolved", "mean"), 本文長=("body_len", "mean"),
        suggestion=("has_suggestion", "mean")).rename(index={0.0: "人間", 1.0: "AI"})
    print(g.to_string(float_format=lambda x: f"{x:10.4f}"))

    out: dict = {"n_threads": int(len(df)),
                 "descriptive": {str(k): {c: float(v) for c, v in r.items()}
                                 for k, r in g.iterrows()}}

    import statsmodels.formula.api as smf
    df["log_len"] = np.log1p(df["body_len"])
    df["pid"] = df["repo"] + "#" + df["pr_number"].astype(str)

    print("\n=== A. スレッド単位の回帰(被説明変数 = isResolved)===")
    print("  PR 固定効果は within 変換で吸収。標準誤差は repo クラスタ。")
    specs = [
        ("resolved ~ is_ai", "統制なし"),
        ("resolved ~ is_ai + log_len + has_suggestion + has_code_block", "内容の特徴を統制"),
        ("resolved ~ is_ai + log_len + has_suggestion + has_code_block + is_test + is_doc + is_config + path_depth",
         "+ ファイル種別"),
    ]
    out["models_pooled"] = {}
    for f, lab in specs:
        m = smf.ols(f, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["repo"]})
        print(f"  {lab:22s} is_ai {m.params['is_ai']:+.4f} "
              f"(se={m.bse['is_ai']:.4f}, p={m.pvalues['is_ai']:.4g})")
        out["models_pooled"][lab] = {"coef": float(m.params["is_ai"]),
                                     "se": float(m.bse["is_ai"]),
                                     "p": float(m.pvalues["is_ai"]), "n": int(m.nobs)}

    # PR 固定効果: within 変換
    cols = ["resolved", "is_ai", "log_len", "has_suggestion", "has_code_block",
            "is_test", "is_doc", "is_config", "path_depth"]
    w = df.copy()
    for c in cols:
        w[c + "_dm"] = w[c] - w.groupby("pid")[c].transform("mean")
    # PR 内に両カテゴリが居ないと is_ai の within 変動がないので落ちる
    mixed = w.groupby("pid")["is_ai"].transform("nunique") > 1
    wm = w[mixed]
    print(f"\n  両方が居る PR に限定: {wm['pid'].nunique():,} PR / {len(wm):,} スレッド")
    for f, lab in [
        ("resolved_dm ~ is_ai_dm - 1", "PR固定効果のみ"),
        ("resolved_dm ~ is_ai_dm + log_len_dm + has_suggestion_dm + has_code_block_dm - 1",
         "PR固定効果 + 内容"),
        ("resolved_dm ~ is_ai_dm + log_len_dm + has_suggestion_dm + has_code_block_dm"
         " + is_test_dm + is_doc_dm + is_config_dm + path_depth_dm - 1",
         "PR固定効果 + 内容 + 種別"),
    ]:
        m = smf.ols(f, data=wm).fit(cov_type="cluster", cov_kwds={"groups": wm["repo"]})
        star = " *" if m.pvalues["is_ai_dm"] < 0.05 else ""
        print(f"  {lab:22s} is_ai {m.params['is_ai_dm']:+.4f} "
              f"(se={m.bse['is_ai_dm']:.4f}, p={m.pvalues['is_ai_dm']:.4g}){star}")
        out.setdefault("models_prfe", {})[lab] = {
            "coef": float(m.params["is_ai_dm"]), "se": float(m.bse["is_ai_dm"]),
            "p": float(m.pvalues["is_ai_dm"]), "n": int(m.nobs)}

    # --- B. 同一箇所の対 ---
    print("\n=== B. ★同一箇所の対(同じ PR・同じ path・line±3 に両方が居る)===")
    dfp = add_pair_flags(df)
    pair = dfp[dfp["has_counterpart"] == 1]
    print(f"  対になったスレッド {len(pair):,} 件 "
          f"(AI {int((pair.is_ai==1).sum()):,} / 人間 {int((pair.is_ai==0).sum()):,})")
    if len(pair) > 100:
        gp = pair.groupby("is_ai").agg(
            n=("outdated", "size"), outdated=("outdated", "mean"),
            resolved=("resolved", "mean")).rename(index={0.0: "人間", 1.0: "AI"})
        print(gp.to_string(float_format=lambda x: f"{x:10.4f}"))
        out["pairs"] = {str(k): {c: float(v) for c, v in r.items()} for k, r in gp.iterrows()}
        for y in ["resolved"]:
            m = smf.ols(f"{y} ~ is_ai + log_len + has_suggestion + has_code_block",
                        data=pair).fit(cov_type="cluster", cov_kwds={"groups": pair["repo"]})
            star = " *" if m.pvalues["is_ai"] < 0.05 else ""
            print(f"  {y:10s} is_ai {m.params['is_ai']:+.4f} "
                  f"(p={m.pvalues['is_ai']:.4g}) n={int(m.nobs):,}{star}")

        print("\n  対になっているか否かで AI スレッドの帰趨は変わるか")
        ai = dfp[dfp["is_ai"] == 1]
        ga = ai.groupby("has_counterpart").agg(
            n=("outdated", "size"), outdated=("outdated", "mean"),
            resolved=("resolved", "mean")).rename(index={0.0: "AI のみ", 1.0: "人間も同じ箇所"})
        print(ga.to_string(float_format=lambda x: f"{x:10.4f}"))
        out["ai_by_counterpart"] = {str(k): {c: float(v) for c, v in r.items()}
                                    for k, r in ga.iterrows()}

    # ---- 対抗仮説の検定 ----
    print("\n=== 対抗仮説の検定 ===")
    pair = dfp[dfp["has_counterpart"] == 1].copy()
    pair["log_len"] = np.log1p(pair["body_len"])
    pair["pid"] = pair["repo"] + "#" + pair["pr_number"].astype(str)
    for c in ["resolved", "outdated", "is_ai", "log_len", "has_suggestion", "has_code_block"]:
        pair[c + "_dm"] = pair[c] - pair.groupby("pid")[c].transform("mean")

    print("\n  [1] AI ツールが自分のスレッドを自動解決しているだけか")
    ai_all = dfp[dfp["is_ai"] == 1]
    ga = ai_all.groupby("starter").agg(n=("resolved", "size"), resolved=("resolved", "mean"),
                                       repos=("repo", "nunique")).sort_values("n", ascending=False)
    print(ga.head(8).to_string(float_format=lambda x: f"{x:9.4f}"))
    print(f"    (参考) 人間 n={int((dfp.is_ai==0).sum()):,} resolved="
          f"{dfp[dfp.is_ai==0]['resolved'].mean():.4f}")
    print("    → 1.0 に張り付くエージェントが無いので、単一ツールの自動解決では説明できない")
    out["by_agent_resolution"] = {k: {"n": int(v["n"]), "resolved": float(v["resolved"]),
                                      "repos": int(v["repos"])} for k, v in ga.iterrows()}

    print("\n  [2] リポジトリの構成差ではないか(対が200件以上のリポジトリ個別)")
    rows = []
    for repo, g2 in pair.groupby("repo"):
        if len(g2) < 200 or g2["is_ai"].nunique() < 2:
            continue
        m = smf.ols("resolved_dm ~ is_ai_dm - 1", data=g2).fit(cov_type="HC1")
        rows.append({"repo": repo, "coef": float(m.params["is_ai_dm"]),
                     "p": float(m.pvalues["is_ai_dm"]), "n": int(m.nobs)})
        star = " *" if m.pvalues["is_ai_dm"] < 0.05 else ""
        print(f"    {repo:42s} {m.params['is_ai_dm']:+.4f} "
              f"(p={m.pvalues['is_ai_dm']:.3g}) n={int(m.nobs):,}{star}")
    print(f"    → {sum(1 for r in rows if r['coef'] > 0)}/{len(rows)} リポジトリで正")
    out["by_repo"] = rows

    print("\n  [3] 人間のは議論だから閉じにくいだけか(本文長の四分位で層別)")
    pair["len_bin"] = pd.qcut(pair["body_len"], 4, labels=["短", "中短", "中長", "長"],
                              duplicates="drop")
    g3 = pair.groupby(["len_bin", "is_ai"], observed=True)["resolved"].mean().unstack()
    g3.columns = ["人間", "AI"]
    print(g3.to_string(float_format=lambda x: f"{x:9.4f}"))
    print("    → 全ての長さ帯で AI が上。長さでは説明できない")
    out["by_length"] = {str(k): {c: float(v) for c, v in r.items()} for k, r in g3.iterrows()}

    print("\n  [4] 参考: この標本は 92% が outdated=0(コード未変更)である")
    for od, lab in [(0.0, "コード変化なし"), (1.0, "コード変化あり")]:
        sub = pair[pair["outdated"] == od]
        if len(sub) < 200:
            continue
        m = smf.ols("resolved_dm ~ is_ai_dm + log_len_dm + has_suggestion_dm - 1",
                    data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["repo"]})
        star = " *" if m.pvalues["is_ai_dm"] < 0.05 else ""
        print(f"    {lab:12s} is_ai {m.params['is_ai_dm']:+.4f} "
              f"(p={m.pvalues['is_ai_dm']:.4g}) n={int(m.nobs):,}{star}")
        out.setdefault("by_outdated", {})[lab] = {
            "coef": float(m.params["is_ai_dm"]), "p": float(m.pvalues["is_ai_dm"]),
            "n": int(m.nobs)}

    (OUT_DIR / "d36_who_said_it.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n-> {OUT_DIR / 'd36_who_said_it.json'}")


if __name__ == "__main__":
    main()

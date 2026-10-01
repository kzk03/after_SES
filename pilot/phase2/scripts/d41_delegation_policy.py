"""D41: 委譲方策の仕様と価値 — 「適用可能 ∩ 人間も支持」を切り出す。

本研究はここまで「AI を入れても人間の作業は減らない」「束縛条件は能力ではなく権限」を
示してきた。では**何を委譲すれば実際に人間の作業が減るのか**。
処方を出さない限り、この研究は記述で終わる。

委譲の対象は2つの条件を同時に満たす指摘である。

  適用可能  機械が適用できる = GitHub の ``` suggestion ブロックを持つ
  人間も支持 人間も独立に同じ箇所を指摘した = その指摘は人間基準でも妥当だった

D40 で分かったのは、この2つが**逆相関する**ことである
(suggestion 形式は実施率 +13〜17pt だが人間の同意率 −5〜8pt)。
だから「suggestion を自動適用せよ」は処方にならない。交わりを狙う必要がある。

本スクリプトは3段階で処方を作る。

  1. **上限**: 完璧な方策でも取り除ける人間のコメントは最大どれだけか
  2. **予測**: 投稿時点の情報だけで「人間も指摘するか」を当てられるか(時系列分割)
  3. **方策曲線**: 閾値ごとに(節約できる人間のコメント数, 人間の支持なしに適用される件数)

出力: outputs_full/d41_delegation_policy.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d41_delegation_policy.py
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name

TOL = 3
TEST_PAT = re.compile(r"(^|/)(tests?|spec|e2e|__tests__)(/|$)|_test\.|\.test\.", re.I)
DOC_PAT = re.compile(r"(^|/)(docs?|website|examples?)(/|$)|\.md$|\.rst$", re.I)
CFG_PAT = re.compile(r"(^|/)(\.github|ci|config|build|scripts)(/|$)|\.ya?ml$|\.json$|\.toml$", re.I)


def load_comments() -> pd.DataFrame:
    """review_bodies から、位置を持つコメントを人間 / AI 両方について取り出す。"""
    rows = []
    for repo in load_repos():
        bpath = RAW_DIR / repo_dir_name(repo) / "review_bodies.jsonl"
        apath = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not (bpath.exists() and apath.exists()):
            continue
        a = pd.read_parquet(apath)
        cat = dict(zip(a["actor_login"], a["category"]))
        with bpath.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                num = d.get("number")
                for ri, rv in enumerate(d.get("reviews") or []):
                    au = rv.get("author")
                    if not au:
                        continue
                    # review_bodies の author は login の文字列(dict のこともある)
                    login = au["login"] if isinstance(au, dict) else au
                    c = cat.get(login, "unknown")
                    if c not in ("human", "agent"):
                        continue
                    for cm in rv.get("comments") or []:
                        ln = cm.get("line") or cm.get("originalLine")
                        if ln is None:
                            continue
                        body = cm.get("body") or ""
                        path = cm.get("path") or ""
                        rows.append({
                            "repo": repo, "pr_number": num, "path": path, "line": int(ln),
                            "cat": c, "is_ai": float(c == "agent"),
                            "login": login, "review_idx": ri,
                            "submitted_at": rv.get("submittedAt"),
                            "body_len": len(body),
                            "has_suggestion": float("```suggestion" in body),
                            "has_code_block": float("```" in body),
                            "has_severity": float(bool(re.search(
                                r"\b(critical|high|medium|low|severity|warning|error)\b", body, re.I))),
                            "n_lines": body.count("\n") + 1,
                            "is_test": float(bool(TEST_PAT.search(path))),
                            "is_doc": float(bool(DOC_PAT.search(path))),
                            "is_config": float(bool(CFG_PAT.search(path))),
                            "path_depth": path.count("/"),
                        })
    return pd.DataFrame(rows)


def mark_counterpart(df: pd.DataFrame) -> pd.DataFrame:
    """各コメントについて、同じ PR・同じ path の line±TOL に
    (a) 反対カテゴリのコメントがあるか (b) suggestion 付き AI コメントがあるか。"""
    n = len(df)
    has_other = np.zeros(n)
    has_ai_sug = np.zeros(n)
    pos = {ix: i for i, ix in enumerate(df.index)}
    for _, g in df.groupby(["repo", "pr_number", "path"], sort=False):
        ai = g[g["is_ai"] == 1]
        hu = g[g["is_ai"] == 0]
        ai_lines = ai["line"].to_numpy()
        hu_lines = hu["line"].to_numpy()
        ai_sug_lines = ai.loc[ai["has_suggestion"] == 1, "line"].to_numpy()
        for ix, ln, isai in zip(g.index, g["line"], g["is_ai"]):
            other = hu_lines if isai == 1 else ai_lines
            if len(other) and np.any(np.abs(other - ln) <= TOL):
                has_other[pos[ix]] = 1.0
            if isai == 0 and len(ai_sug_lines) and np.any(np.abs(ai_sug_lines - ln) <= TOL):
                has_ai_sug[pos[ix]] = 1.0
    df = df.copy()
    df["human_same_spot"] = has_other          # AI 行なら「人間も指摘した」
    df["covered_by_ai_suggestion"] = has_ai_sug  # 人間行なら「suggestion 付き AI が同じ箇所」
    return df


def main() -> None:
    df = load_comments()
    print(f"=== 位置付きコメント {len(df):,} 件 "
          f"(人間 {int((df.is_ai==0).sum()):,} / AI {int(df.is_ai.sum()):,}) "
          f"/ PR {df.groupby(['repo','pr_number']).ngroups:,} ===")
    df = mark_counterpart(df)
    hu = df[df["is_ai"] == 0]
    ai = df[df["is_ai"] == 1]

    out: dict = {"n_comments": int(len(df)), "n_human": int(len(hu)), "n_ai": int(len(ai))}

    print("\n" + "=" * 66)
    print("1. 上限 — 完璧な自動適用でも取り除ける人間のコメントは最大どれだけか")
    print("=" * 66)
    any_ai = hu["human_same_spot"].mean()
    by_sug = hu["covered_by_ai_suggestion"].mean()
    print(f"  人間の位置付きコメント {len(hu):,} 件のうち")
    print(f"    AI も同じ箇所を指摘していた            {any_ai:7.2%}")
    print(f"    ★suggestion 付き AI が同じ箇所を指摘   {by_sug:7.2%}  ← 自動適用の上限")
    print(f"\n  → 完璧な方策でも、人間のレビューコメントの {by_sug:.1%} しか取り除けない。")
    print(f"     残り {1-by_sug:.1%} は機械が適用できる形で指摘されていない。")
    out["ceiling"] = {"human_comments": int(len(hu)),
                      "covered_by_any_ai": float(any_ai),
                      "covered_by_ai_suggestion": float(by_sug)}

    print("\n  内訳(ファイル種別ごとの上限):")
    hu2 = hu.copy()
    hu2["種別"] = np.select([hu2.is_test == 1, hu2.is_doc == 1, hu2.is_config == 1],
                           ["テスト", "ドキュメント", "設定"], default="コード")
    print(hu2.groupby("種別")["covered_by_ai_suggestion"].agg(["size", "mean"])
          .to_string(float_format=lambda x: f"{x:10.4f}"))

    print("\n" + "=" * 66)
    print("2. 予測 — 投稿時点の情報で『人間も指摘するか』を当てられるか")
    print("=" * 66)
    sug = ai[ai["has_suggestion"] == 1].dropna(subset=["submitted_at"]).copy()
    sug["t"] = pd.to_datetime(sug["submitted_at"], utc=True, errors="coerce")
    sug = sug.dropna(subset=["t"]).sort_values("t")
    print(f"  対象: suggestion 付き AI コメント {len(sug):,} 件")
    print(f"  うち人間も同じ箇所を指摘した割合(基準率): {sug.human_same_spot.mean():.2%}")
    if len(sug) < 500:
        print("  件数が足りないので予測は行わない")
        (OUT_DIR / "d41_delegation_policy.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2))
        return

    cut = sug["t"].quantile(0.7)
    tr, te = sug[sug["t"] <= cut], sug[sug["t"] > cut]
    feats = ["body_len", "n_lines", "has_code_block", "has_severity",
             "is_test", "is_doc", "is_config", "path_depth", "review_idx"]
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr = tr[feats].fillna(0).to_numpy(), tr["human_same_spot"].to_numpy()
    Xte, yte = te[feats].fillna(0).to_numpy(), te["human_same_spot"].to_numpy()
    print(f"  学習 {len(tr):,} 件(〜{cut.date()}) / 検証 {len(te):,} 件")
    out["prediction"] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                         "base_rate": float(sug["human_same_spot"].mean())}
    sc = StandardScaler().fit(Xtr)
    models = {"ロジスティック回帰": LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr),
              "勾配ブースティング": GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)}
    best_name, best_p, best_auc = None, None, -1
    for name, m in models.items():
        pr = m.predict_proba(sc.transform(Xte) if "ロジ" in name else Xte)[:, 1]
        auc = roc_auc_score(yte, pr)
        print(f"    {name:16s} AUC={auc:.4f}")
        out["prediction"][name] = {"auc": float(auc)}
        if auc > best_auc:
            best_name, best_p, best_auc = name, pr, auc

    print("\n" + "=" * 66)
    print("3. 方策曲線 — 上位 k% を自動適用したら何が起きるか")
    print("=" * 66)
    base = yte.mean()
    print(f"  検証集合 {len(te):,} 件 / 基準率 {base:.2%}(無作為に選んだときの精度)")
    print(f"  {'上位':>6s} {'適用件数':>8s} {'精度':>8s} {'回収率':>8s} {'支持なし適用':>10s}")
    rows = []
    order = np.argsort(-best_p)
    for k in [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]:
        n_take = max(1, int(len(te) * k))
        idx = order[:n_take]
        prec = yte[idx].mean()
        rec = yte[idx].sum() / max(yte.sum(), 1)
        wrong = int((1 - yte[idx]).sum())
        print(f"  {k:6.0%} {n_take:8,} {prec:8.2%} {rec:8.2%} {wrong:10,}")
        rows.append({"k": k, "n": int(n_take), "precision": float(prec),
                     "recall": float(rec), "unendorsed": wrong})
    out["policy_curve"] = rows
    lift = rows[1]["precision"] / base if base > 0 else float("nan")
    print(f"\n  上位10%の精度は基準率の {lift:.2f} 倍")

    (OUT_DIR / "d41_delegation_policy.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n-> {OUT_DIR / 'd41_delegation_policy.json'}")


if __name__ == "__main__":
    main()

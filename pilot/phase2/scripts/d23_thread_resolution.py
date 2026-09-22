"""D23: AI の指摘は人間の指摘と同じように処理されているか(スレッド解決率)。

D22 で収集したレビュースレッドの isResolved / isOutdated を使い、
スレッドを開始したのが AI か人間かで処理のされ方を比べる。

交絡に注意が要る。AI がレビューする PR は内生的に選ばれているし、
PR が大きい/小さい、活発/停滞といった性質で解決率は変わる。
そこで **同一 PR 内に AI スレッドと人間スレッドが両方ある部分標本に絞り、PR 固定効果**を入れる。
これで PR の性質を完全に吸収したうえで「同じ PR の中で AI の指摘と人間の指摘がどう扱われたか」を比較できる。

isResolved は「対応した」と「不要と判断した」の両方を含むので、
**「未解決のまま放置された割合」**として解釈するのが安全。
isOutdated(その後のコード変更で該当箇所が古くなった)も併せて見る。

実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d23_thread_resolution.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name


def main() -> None:
    rows = []
    for repo in load_repos():
        path = RAW_DIR / repo_dir_name(repo) / "thread_status.jsonl"
        actors_path = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not path.exists() or not actors_path.exists():
            continue
        a = pd.read_parquet(actors_path)
        cat_of = dict(zip(a["actor_login"], a["category"]))
        with path.open() as f:
            for line in f:
                try:
                    pr = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for th in pr["threads"]:
                    cat = cat_of.get(th["starter"], "unknown")
                    if cat not in ("human", "agent"):
                        continue
                    rows.append({
                        "repo": repo,
                        "pr_number": pr["number"],
                        "starter_cat": cat,
                        "is_agent": 1.0 if cat == "agent" else 0.0,
                        "resolved": 1.0 if th["isResolved"] else 0.0,
                        "outdated": 1.0 if th["isOutdated"] else 0.0,
                        "has_line": 1.0 if th.get("line") is not None else 0.0,
                    })
    if not rows:
        print("解析対象がありません。D22 の収集が進んでいるか確認してください。")
        return

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "d23_threads.parquet", index=False)

    print(f"=== レビュースレッド {len(df):,} 件 "
          f"({df['repo'].nunique()} リポジトリ, {df['pr_number'].nunique():,} PR) ===\n")

    print("【全体】")
    g = df.groupby("starter_cat").agg(
        n=("resolved", "size"), resolved=("resolved", "mean"), outdated=("outdated", "mean"))
    print(f"  {'開始者':<8}{'スレッド数':>12}{'解決率':>10}{'outdated率':>12}")
    for cat, r in g.iterrows():
        print(f"  {cat:<8}{int(r['n']):>12,}{r['resolved']:>10.1%}{r['outdated']:>12.1%}")

    out: dict = {"n_threads": int(len(df)), "overall": g.to_dict(orient="index")}

    # --- 同一 PR 内比較(PR 固定効果) -------------------------------------------
    both = df.groupby(["repo", "pr_number"])["starter_cat"].nunique()
    keys = set(both[both == 2].index)
    sub = df[df.set_index(["repo", "pr_number"]).index.isin(keys)].copy()
    print(f"\n【同一 PR 内比較】AI と人間の両方がスレッドを開始した PR: {len(keys):,} 件, "
          f"スレッド {len(sub):,} 件")
    if len(sub) < 100:
        print("  サンプルが足りません")
    else:
        g2 = sub.groupby("starter_cat").agg(
            n=("resolved", "size"), resolved=("resolved", "mean"), outdated=("outdated", "mean"))
        print(f"  {'開始者':<8}{'スレッド数':>12}{'解決率':>10}{'outdated率':>12}")
        for cat, r in g2.iterrows():
            print(f"  {cat:<8}{int(r['n']):>12,}{r['resolved']:>10.1%}{r['outdated']:>12.1%}")

        sub["pr_key"] = sub["repo"] + "#" + sub["pr_number"].astype(str)
        sub["obs"] = np.arange(len(sub))
        d = sub.set_index(["pr_key", "obs"])
        res = {}
        for y in ["resolved", "outdated"]:
            r = PanelOLS(d[y], d[["is_agent"]], entity_effects=True, drop_absorbed=True).fit(
                cov_type="clustered", clusters=d["repo"])
            res[y] = {"coef": float(r.params["is_agent"]), "se": float(r.std_errors["is_agent"]),
                      "p": float(r.pvalues["is_agent"]), "n": int(r.nobs)}
            print(f"\n  PR 固定効果モデル [{y}] (repoクラスタ, n={int(r.nobs):,})")
            print(f"    is_agent = {r.params['is_agent']:+.4f} "
                  f"(se={r.std_errors['is_agent']:.4f}, p={r.pvalues['is_agent']:.4g})")
        out["pr_fixed_effects"] = res

        c = res["resolved"]["coef"]
        p = res["resolved"]["p"]
        print()
        if p < 0.05 and c < 0:
            print(f"  → 同じ PR の中で、AI が開始したスレッドは人間のものより "
                  f"{abs(c):.1%} ポイント解決されにくい。**補完的な指摘が相対的に軽視されている**")
        elif p < 0.05 and c > 0:
            print(f"  → 同じ PR の中で、AI のスレッドの方が {c:.1%} ポイント解決されやすい")
        else:
            print("  → 同じ PR の中では、AI の指摘も人間の指摘も同じように処理されている")

    # --- 参考: リポジトリ単位の pooled OLS(repo ダミー込み) -----------------------
    dummies = pd.get_dummies(df["repo"], drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([df[["is_agent"]], dummies], axis=1))
    r = sm.OLS(df["resolved"], X).fit(cov_type="cluster", cov_kwds={"groups": df["repo"]})
    out["pooled_with_repo_fe"] = {
        "coef": float(r.params["is_agent"]), "p": float(r.pvalues["is_agent"]), "n": int(r.nobs)}
    print(f"\n【参考】repo 固定効果のみの pooled OLS: is_agent = {r.params['is_agent']:+.4f} "
          f"(p={r.pvalues['is_agent']:.4g}, n={int(r.nobs):,})")

    (OUT_DIR / "d23_thread_resolution.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'd23_thread_resolution.json'}")


if __name__ == "__main__":
    main()

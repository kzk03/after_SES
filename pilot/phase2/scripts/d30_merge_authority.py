"""D30: ★マージ実行権限は機械に開かれているか — 判断の地図を広げる。

D24 は「レビュー承認の回路は機械全体に閉じている」ことを示した。
人間作成 PR の最初の承認者は人間 99.9%、判断率 60.7% の CI ボットですら 0.01%(34件)。

しかしレビュー承認は判断のひとつにすぎない。**マージの実行**はより強い権限である。
承認と同じく人間に閉じているのか、それとも「承認は人間、マージは自動化」という
分業が成立しているのか。もし後者なら、「機械に開かれた判断」が実は存在することになり、
本研究の主張は「承認という特定の行為についてのもの」に限定される。

見るもの:
  1. マージを実行したのは誰か(全体、および PR 作成者のカテゴリ別)
  2. **人間が作成した PR** に限定したときのマージ実行者
  3. 承認者とマージ実行者の関係(承認は人間だがマージは機械、という分業があるか)
  4. AI レビュー導入の前後でマージ実行者の構成が変わるか

実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d30_merge_authority.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "pilot"))
from loop1_actors import classify, load_config  # noqa: E402


def build_classifier():
    """マージ実行者を login から直接分類する。

    actors.parquet はレビュー・PR作成・コミットから作られるので、**マージしかしない
    アカウントは載らない**。マージキュー(trunk-io 等)がまさにそれで、
    actors 経由だと永遠に unknown に落ちる。yaml の規則を直接当てて拾う。
    """
    cfg = load_config()
    agent_accounts = {a.lower() for a in cfg["agent"]["accounts"]}
    legacy_accounts = {a.lower() for a in cfg["legacy_bot"]["accounts"]}
    legacy_patterns = [re.compile(p) for p in cfg["legacy_bot"].get("patterns", [])]

    def classify_merger(login: str | None, cat_of: dict[str, str]) -> str:
        if not login:
            return "none"
        by_rule = classify(login, agent_accounts, legacy_accounts, legacy_patterns)
        if by_rule != "human":
            return by_rule           # yaml が機械と言うならそれが優先
        return cat_of.get(login, "unknown")  # 人間かどうかは actors に載っているかで決める

    return classify_merger


def load_merged_by() -> pd.DataFrame:
    classify_merger = build_classifier()
    rows = []
    for repo in load_repos():
        path = RAW_DIR / repo_dir_name(repo) / "merged_by.jsonl"
        actors_path = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not (path.exists() and actors_path.exists()):
            continue
        a = pd.read_parquet(actors_path)
        cat_of = dict(zip(a["actor_login"], a["category"]))
        with path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mb = d.get("merged_by")
                rows.append({
                    "repo": repo,
                    "pr_number": d["number"],
                    "merged_by": mb,
                    "merger_cat": classify_merger(mb, cat_of),
                })
    return pd.DataFrame(rows)


def main() -> None:
    mb = load_merged_by()
    if mb.empty:
        print("mergedBy のデータがありません。D29 の収集を確認してください。")
        return

    # PR 作成者と承認状況を d24 から引く(なければ d5)
    src = OUT_DIR / "d24_approval.parquet"
    if not src.exists():
        src = OUT_DIR / "d5_pr_level.parquet"
    pr = pd.read_parquet(src)
    keep = [c for c in ["repo", "pr_number", "author_cat", "has_human_appr",
                        "has_agent_appr", "has_legacy_appr", "module_id_global", "month"]
            if c in pr.columns]
    df = mb.merge(pr[keep], on=["repo", "pr_number"], how="left")

    out: dict = {"n": int(len(df)), "n_repos": int(df["repo"].nunique())}
    print(f"=== マージ済み PR {len(df):,} 件({df['repo'].nunique()} リポジトリ) ===")
    print(f"  mergedBy が取得できた割合: {(df['merged_by'].notna()).mean():.1%}\n")

    # --- 1. 全体: 誰がマージしたか -------------------------------------------------
    print("【1】マージを実行した主体(全 PR)")
    vc = df["merger_cat"].value_counts()
    for k, v in vc.items():
        print(f"  {k:<12}{v:>10,}  ({v / len(df):>6.2%})")
    out["merger_overall"] = {str(k): int(v) for k, v in vc.items()}

    # --- 2. PR 作成者のカテゴリ別 ---------------------------------------------------
    if "author_cat" in df.columns:
        print("\n【2】PR 作成者 × マージ実行者")
        ct = pd.crosstab(df["author_cat"], df["merger_cat"], normalize="index")
        n_by = df["author_cat"].value_counts()
        cols = [c for c in ["human", "agent", "legacy_bot", "unknown", "none"] if c in ct.columns]
        print(f"  {'作成者':<12}{'PR数':>10}" + "".join(f"{c:>13}" for c in cols))
        for idx in ct.index:
            if n_by.get(idx, 0) < 100:
                continue
            print(f"  {idx:<12}{int(n_by[idx]):>10,}"
                  + "".join(f"{ct.loc[idx, c]:>13.2%}" for c in cols))
        out["merger_by_author"] = ct.to_dict()

        # --- 3. ★人間作成 PR に限定 ---------------------------------------------
        hp = df[df["author_cat"] == "human"]
        print(f"\n【3】★人間が作成した PR {len(hp):,} 件のマージ実行者")
        vch = hp["merger_cat"].value_counts()
        for k, v in vch.items():
            print(f"  {k:<12}{v:>10,}  ({v / len(hp):>7.3%})")
        out["merger_human_authored"] = {str(k): int(v) for k, v in vch.items()}

        mach = vch.get("agent", 0) + vch.get("legacy_bot", 0)
        print(f"\n  → 機械によるマージ: {mach:,} / {len(hp):,} = {mach / max(len(hp),1):.3%}")
        # 承認側は D24 の出力から読む(手で書くと分類を直したときに古い数字が残る)
        d24_path = OUT_DIR / "d24_approval_substitution.json"
        if d24_path.exists():
            fa = json.loads(d24_path.read_text()).get("first_approver_human_authored", {})
            tot = sum(fa.values())
            m = fa.get("agent", 0) + fa.get("legacy_bot", 0)
            if tot:
                print(f"     (参考) D24 の最初の承認者が機械だった割合: {m / tot:.3%} ({m:,}/{tot:,})")
                out["first_approver_machine_share_d24"] = m / tot

        # --- 4. 承認とマージの分業はあるか ----------------------------------------
        if "has_human_appr" in hp.columns:
            print("\n【4】承認とマージの分業")
            sub = hp[hp["merger_cat"].isin(["human", "agent", "legacy_bot"])]
            tab = pd.crosstab(sub["has_human_appr"].astype(bool), sub["merger_cat"])
            print(f"  {'人間の承認':<14}" + "".join(f"{c:>14}" for c in tab.columns))
            for idx in tab.index:
                lab = "あり" if idx else "なし"
                print(f"  {lab:<14}" + "".join(f"{int(tab.loc[idx, c]):>14,}" for c in tab.columns))
            out["approval_x_merge"] = tab.to_dict()

            # 人間の承認なしで機械がマージした PR = 判断が完全に機械で閉じたケース
            fully = sub[(~sub["has_human_appr"].astype(bool))
                        & (sub["merger_cat"].isin(["agent", "legacy_bot"]))]
            print(f"\n  ★人間の承認もマージも無い(判断が完全に機械で完結した)人間作成 PR: "
                  f"{len(fully):,} 件 ({len(fully) / max(len(hp),1):.3%})")
            out["fully_machine_closed"] = int(len(fully))

    # --- 5. マージ実行者の集中度 ----------------------------------------------------
    print("\n【5】マージ実行権限の集中")
    for repo, g in df.groupby("repo"):
        pass
    conc = []
    for repo, g in df[df["merged_by"].notna()].groupby("repo"):
        vcm = g["merged_by"].value_counts()
        if len(vcm) < 2:
            continue
        conc.append({"repo": repo, "n_mergers": len(vcm),
                     "top1_share": float(vcm.iloc[0] / vcm.sum()),
                     "top10pct_share": float(
                         vcm.head(max(1, len(vcm) // 10)).sum() / vcm.sum())})
    cdf = pd.DataFrame(conc)
    if len(cdf):
        print(f"  マージ実行者の人数(中央値): {cdf['n_mergers'].median():.0f}")
        print(f"  上位1人のシェア(中央値)   : {cdf['top1_share'].median():.1%}")
        print(f"  上位10%のシェア(中央値)   : {cdf['top10pct_share'].median():.1%}")
        out["merge_concentration"] = {
            "median_n_mergers": float(cdf["n_mergers"].median()),
            "median_top1_share": float(cdf["top1_share"].median()),
            "median_top10pct_share": float(cdf["top10pct_share"].median()),
        }
        # 手書きしていた 73.5% は古い値だった(2026-09-27 査読 R4)。D15 の出力から読む
        d15p = OUT_DIR / "d15_burden_distribution.json"
        if d15p.exists():
            d15 = json.loads(d15p.read_text())
            v = d15.get("top10pct_share") or d15.get("descriptive", {}).get("top10pct_share")
            if v is not None:
                print(f"  (参考) レビュー負荷は上位10%が{v:.1%}を担っている(D15)")

    (OUT_DIR / "d30_merge_authority.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'd30_merge_authority.json'}")


if __name__ == "__main__":
    main()

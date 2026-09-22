"""D26: ★実務価値の検証 — 「読むべき AI コメント」を予測できるか。

本研究の発見のうち、現場に最も直接効くのはこれである。

  - AI レビュー導入後、人間が承認前に読む AI コメントは **+48%** 増える(D7)
  - AI コメント1件は人間コメント **4.5件分**の読む量(D21)
  - **しかし AI の指摘が実際にコード変更につながる割合は低い**
    (先行研究 arXiv 2508.18771 で 0.9〜19.2%、我々の outdated 率でも AI 48.4% vs 人間 61.9%)

つまり **増えた読む量の大半が、コード変更につながらない**。
ここに明確な無駄がある。「どの AI コメントを読むべきか」を事前に見分けられれば、
レビュアーの読む量を直接削減できる。

本スクリプトはその実現可能性を検証する。

  目的変数 : isOutdated(そのスレッドの該当箇所がその後変更された = 対処された代理指標)
             ※ isResolved は「不要と判断して閉じた」を含むので使わない(D23 の教訓)
  特徴量   : コメントの長さ・構造、重大度ラベル、ファイル種別、
             同じ箇所を人間も指摘しているか、PR 規模、エージェント種別
  評価     : 時系列分割(過去で学習 → 未来でテスト)。AUC に加えて
             **「上位 k% を読めば、対処される指摘の何 % をカバーできるか」**を報告する。
             これが読む量削減の ROI に直結する実務指標。

実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d26_triage_model.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name

HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
SEVERITY = re.compile(r"\b(critical|high severity|blocker|security|vulnerab)", re.I)
CODE_BLOCK = re.compile(r"```")
SUGGESTION = re.compile(r"```suggestion")


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_dataset() -> pd.DataFrame:
    rows = []
    for repo in load_repos():
        d = RAW_DIR / repo_dir_name(repo)
        th_path, bd_path = d / "thread_status.jsonl", d / "review_bodies.jsonl"
        actors_path = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not (th_path.exists() and bd_path.exists() and actors_path.exists()):
            continue
        a = pd.read_parquet(actors_path)
        cat_of = dict(zip(a["actor_login"], a["category"]))

        # スレッド: (pr, path, line) -> 状態
        threads: dict[tuple, dict] = {}
        for rec in read_jsonl(th_path):
            for t in rec["threads"]:
                threads[(rec["number"], t["path"], t["line"])] = t

        for rec in read_jsonl(bd_path):
            pr_no = rec["number"]
            # この PR で人間が指摘した箇所(±3行に丸め)
            human_locs = set()
            n_ai_comments = n_human_comments = 0
            for rv in rec["reviews"]:
                cat = cat_of.get(rv["author"], "unknown")
                for c in rv["comments"]:
                    if cat == "human":
                        n_human_comments += 1
                        hl = c.get("originalLine") if c.get("originalLine") is not None else c.get("line")
                        if c.get("path") and hl is not None:
                            human_locs.add((c["path"], int(hl) // 7))
                    elif cat == "agent":
                        n_ai_comments += 1

            for idx, rv in enumerate(rec["reviews"]):
                if cat_of.get(rv["author"]) != "agent":
                    continue
                for c in rv["comments"]:
                    key = (pr_no, c.get("path"), c.get("line"))
                    th = threads.get(key)
                    if th is None:
                        continue
                    # ★ line は outdated になると GitHub 側で null になるため、
                    #   has_line や line ベースの特徴量は目的変数のリークになる。
                    #   位置は originalLine(outdated でも残る)で取る。
                    line_stable = c.get("originalLine")
                    if line_stable is None:
                        line_stable = c.get("line")
                    body = c.get("body") or ""
                    visible = HTML_COMMENT.sub("", body)
                    path = c.get("path") or ""
                    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
                    loc = (path, int(line_stable) // 7) if line_stable is not None else None
                    rows.append({
                        "repo": repo,
                        "pr_number": pr_no,
                        "agent": rv["author"],
                        "created_at": th.get("createdAt"),
                        # --- 目的変数 ---
                        "outdated": 1 if th.get("isOutdated") else 0,
                        "resolved": 1 if th.get("isResolved") else 0,
                        # --- 特徴量 ---
                        "len_visible": len(visible),
                        "n_lines": visible.count("\n") + 1,
                        "has_severity": 1 if SEVERITY.search(visible) else 0,
                        "has_code_block": 1 if CODE_BLOCK.search(visible) else 0,
                        "has_suggestion": 1 if SUGGESTION.search(body) else 0,
                        "is_test_file": 1 if re.search(r"(^|/)(test|tests|spec)", path, re.I) else 0,
                        "is_config": 1 if ext in {"json", "yaml", "yml", "toml", "cfg", "ini", "lock"} else 0,
                        "is_doc": 1 if ext in {"md", "rst", "txt"} else 0,
                        "path_depth": path.count("/"),
                        "human_same_spot": 1 if (loc is not None and loc in human_locs) else 0,
                        "n_human_comments_in_pr": n_human_comments,
                        "n_ai_comments_in_pr": n_ai_comments,
                        "review_idx": idx,
                    })
    return pd.DataFrame(rows)


FEATURES = ["len_visible", "n_lines", "has_severity", "has_code_block", "has_suggestion",
            "is_test_file", "is_config", "is_doc", "path_depth",
            "human_same_spot", "n_human_comments_in_pr", "n_ai_comments_in_pr", "review_idx"]


def recall_at_k(y_true: np.ndarray, score: np.ndarray, k: float) -> float:
    """スコア上位 k(割合)を読んだとき、対処される指摘の何割をカバーできるか。"""
    n = max(int(len(score) * k), 1)
    idx = np.argsort(-score)[:n]
    return float(y_true[idx].sum() / max(y_true.sum(), 1))


def main() -> None:
    df = build_dataset()
    if df.empty:
        print("データがありません。D20/D22 の収集を確認してください。")
        return
    df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "d26_triage.parquet", index=False)

    print(f"=== AI レビューコメント {len(df):,} 件({df['repo'].nunique()} リポジトリ) ===")
    print(f"  期間: {df['created_at'].min()[:7]} 〜 {df['created_at'].max()[:7]}")
    print(f"  対処された(outdated=1)割合: {df['outdated'].mean():.1%}")
    print(f"  参考: resolved=1 の割合: {df['resolved'].mean():.1%}\n")

    # 時系列分割: 前半70%で学習、後半30%でテスト
    cut = int(len(df) * 0.7)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    print(f"学習 {len(tr):,} 件({tr['created_at'].min()[:7]}〜{tr['created_at'].max()[:7]}) / "
          f"テスト {len(te):,} 件({te['created_at'].min()[:7]}〜{te['created_at'].max()[:7]})\n")

    y_tr, y_te = tr["outdated"].values, te["outdated"].values
    out: dict = {"n": int(len(df)), "base_rate": float(df["outdated"].mean()),
                 "n_train": int(len(tr)), "n_test": int(len(te)), "models": {}}

    print(f"{'モデル':<22}{'AUC':>8}{'上位10%':>10}{'上位20%':>10}{'上位30%':>10}{'上位50%':>10}")
    print("-" * 70)
    # ベースライン: ランダム(= 読んだ割合ぶんだけカバーできる)
    print(f"{'ランダム(baseline)':<22}{0.5:>8.3f}{0.10:>10.1%}{0.20:>10.1%}{0.30:>10.1%}{0.50:>10.1%}")

    for name, make in [
        ("ロジスティック回帰", "lr"),
        ("勾配ブースティング", "gb"),
    ]:
        if make == "lr":
            sc = StandardScaler().fit(tr[FEATURES])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(tr[FEATURES]), y_tr)
            score = clf.predict_proba(sc.transform(te[FEATURES]))[:, 1]
            coef = dict(zip(FEATURES, clf.coef_[0].tolist()))
        else:
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
            clf.fit(tr[FEATURES], y_tr)
            score = clf.predict_proba(te[FEATURES])[:, 1]
            coef = None
        auc = roc_auc_score(y_te, score)
        rk = {k: recall_at_k(y_te, score, k) for k in (0.1, 0.2, 0.3, 0.5)}
        out["models"][name] = {"auc": float(auc), "recall_at_k": rk, "coef": coef}
        print(f"{name:<22}{auc:>8.3f}{rk[0.1]:>10.1%}{rk[0.2]:>10.1%}{rk[0.3]:>10.1%}{rk[0.5]:>10.1%}")

    # 実務的な言い換え
    best = max(out["models"].items(), key=lambda kv: kv[1]["auc"])
    r30 = best[1]["recall_at_k"][0.3]
    print(f"\n★ {best[0]} の場合: **上位30%だけ読めば、対処される指摘の {r30:.0%} をカバーできる**")
    print(f"   → 読む量を 70% 削減して、拾える指摘は {r30:.0%} 維持。"
          f"ランダムに30%読む場合({0.30:.0%})の {r30 / 0.30:.1f}倍の効率")

    lr = out["models"].get("ロジスティック回帰", {}).get("coef")
    if lr:
        print("\n【何が「対処される指摘」を予測するか】(ロジスティック回帰の標準化係数)")
        for k, v in sorted(lr.items(), key=lambda kv: -abs(kv[1]))[:8]:
            print(f"  {k:<26}{v:+.3f}")

    # エージェント別の対処率(ツール選定の参考になる)
    ag = df.groupby("agent").agg(n=("outdated", "size"), outdated=("outdated", "mean"),
                                 len_visible=("len_visible", "mean"))
    ag = ag[ag["n"] >= 100].sort_values("n", ascending=False)
    if len(ag):
        print(f"\n【エージェント別】{'コメント数':>12}{'対処率':>10}{'平均文字数':>12}")
        for login, r in ag.iterrows():
            print(f"  {login:<28}{int(r['n']):>10,}{r['outdated']:>10.1%}{r['len_visible']:>12.0f}")
        out["by_agent"] = ag.to_dict(orient="index")

    (OUT_DIR / "d26_triage_model.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'd26_triage_model.json'}")


if __name__ == "__main__":
    main()

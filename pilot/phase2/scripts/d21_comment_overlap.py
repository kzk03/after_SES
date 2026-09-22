"""D21: AI と人間は同じ箇所を指摘しているのか — 冗長性の直接証拠。

D18 は「AI レビューが付いた PR の 93% は人間も見ている」を示したが、これは **PR 単位**の話で、
同じ PR の中で AI と人間が別々の箇所を見ている可能性を排除できていない。
D20 で収集したコメントの位置(path, line)を使い、**行単位**で重なりを測る。

  - 重なりが高い → AI は人間が既に気づく箇所を繰り返している(冗長。読む量だけが増える)
  - 重なりが低い → AI は人間が見落とす箇所を拾っている(補完。人間が減らなくても価値がある)

比較対照が要る。「AI と人間」の重なりだけ見ても高いか低いか判断できないので、
**同じ PR の中の「人間どうし」の重なり**を基準にする。
人間レビュアー2人の重なりと同程度なら「AI は人間レビュアーの1人と同じように振る舞っている」、
それより明確に高ければ「AI は冗長」と言える。

指標:
  file_overlap : 指摘したファイル集合の Jaccard
  line_overlap : (path, line) を ±TOL 行の粒度に丸めた集合の Jaccard
  ai_covered   : AI の指摘のうち人間も指摘した箇所の割合
  who_first    : 同じ箇所を両者が指摘した場合、どちらが先か

実行: uv run python pilot/phase2/scripts/d21_comment_overlap.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name

TOL = 3  # 何行以内を「同じ箇所」とみなすか
# AI のコメントには LOCATIONS や BUG_ID などの HTML コメントが埋め込まれていて画面には出ない。
# 「人間が実際に読む量」を測るので、これを除いた文字数も併せて数える
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def parse_dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def loc_key(c: dict) -> tuple[str, int] | None:
    """コメント位置を (path, 丸めた行) に変換する。"""
    path = c.get("path")
    line = c.get("line") or c.get("originalLine")
    if not path or line is None:
        return None
    return (path, int(line) // (2 * TOL + 1))


def jaccard(a: set, b: set) -> float | None:
    if not a or not b:
        return None
    return len(a & b) / len(a | b)


def main() -> None:
    rows = []
    for repo in load_repos():
        d = RAW_DIR / repo_dir_name(repo)
        bodies = d / "review_bodies.jsonl"
        actors_path = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not bodies.exists() or not actors_path.exists():
            continue
        cat_of = dict(zip(*pd.read_parquet(actors_path)[["actor_login", "category"]].values.T))

        with bodies.open() as f:
            for line in f:
                try:
                    pr = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 担い手別に「指摘した箇所」と「最初に指摘した時刻」を集める
                locs: dict[str, set] = {"human": set(), "agent": set()}
                files: dict[str, set] = {"human": set(), "agent": set()}
                first_at: dict[tuple, dict[str, datetime]] = {}
                n_comments = {"human": 0, "agent": 0}
                n_chars = {"human": 0, "agent": 0}
                n_chars_visible = {"human": 0, "agent": 0}
                humans_locs: dict[str, set] = {}

                for rv in pr["reviews"]:
                    cat = cat_of.get(rv["author"], "unknown")
                    if cat not in ("human", "agent"):
                        continue
                    t = parse_dt(rv.get("submittedAt"))
                    for c in rv["comments"]:
                        k = loc_key(c)
                        n_comments[cat] += 1
                        body = c.get("body") or ""
                        n_chars[cat] += len(body)
                        n_chars_visible[cat] += len(HTML_COMMENT.sub("", body))
                        if k is None:
                            continue
                        locs[cat].add(k)
                        files[cat].add(k[0])
                        if t is not None:
                            cur = first_at.setdefault(k, {})
                            if cat not in cur or t < cur[cat]:
                                cur[cat] = t
                        if cat == "human":
                            humans_locs.setdefault(rv["author"], set()).add(k)

                if not locs["agent"] or not locs["human"]:
                    # 片方しか位置付きコメントを出していない PR は重なりを定義できない
                    continue

                both = locs["agent"] & locs["human"]
                ai_first = sum(
                    1 for k in both
                    if "agent" in first_at.get(k, {}) and "human" in first_at.get(k, {})
                    and first_at[k]["agent"] < first_at[k]["human"]
                )
                n_both_timed = sum(
                    1 for k in both
                    if "agent" in first_at.get(k, {}) and "human" in first_at.get(k, {})
                )

                # 対照: 同じ PR 内の人間どうしの重なり(2人以上いる場合)
                hh = None
                hs = [s for s in humans_locs.values() if s]
                if len(hs) >= 2:
                    vals = [jaccard(hs[i], hs[j]) for i in range(len(hs)) for j in range(i + 1, len(hs))]
                    vals = [v for v in vals if v is not None]
                    hh = float(np.mean(vals)) if vals else None

                rows.append({
                    "repo": repo,
                    "pr_number": pr["number"],
                    "n_ai_locs": len(locs["agent"]),
                    "n_human_locs": len(locs["human"]),
                    "file_overlap": jaccard(files["agent"], files["human"]),
                    "line_overlap": jaccard(locs["agent"], locs["human"]),
                    "ai_covered_by_human": len(both) / len(locs["agent"]),
                    "human_covered_by_ai": len(both) / len(locs["human"]),
                    "human_human_overlap": hh,
                    "n_both_timed": n_both_timed,
                    "ai_first_share": (ai_first / n_both_timed) if n_both_timed else None,
                    "ai_comments": n_comments["agent"],
                    "human_comments": n_comments["human"],
                    "ai_chars": n_chars["agent"],
                    "human_chars": n_chars["human"],
                    "ai_chars_visible": n_chars_visible["agent"],
                    "human_chars_visible": n_chars_visible["human"],
                })

    if not rows:
        print("解析対象がありません。D20 の収集が完了しているか確認してください。")
        return
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "d21_overlap.parquet", index=False)

    print(f"=== AI と人間が両方とも位置付きコメントを出した PR: {len(df):,} 件 "
          f"({df['repo'].nunique()} リポジトリ) ===\n")

    def show(col: str, label: str) -> None:
        s = df[col].dropna()
        if s.empty:
            print(f"  {label:<34} (データなし)")
            return
        print(f"  {label:<34} 平均 {s.mean():.3f}  中央値 {s.median():.3f}  n={len(s):,}")

    print("【重なり】")
    show("file_overlap", "ファイル単位 Jaccard (AI×人間)")
    show("line_overlap", f"行単位 Jaccard (±{TOL}行, AI×人間)")
    show("human_human_overlap", "行単位 Jaccard (人間×人間) ★対照")
    show("ai_covered_by_human", "AI の指摘のうち人間も指摘した割合")
    show("human_covered_by_ai", "人間の指摘のうち AI も指摘した割合")

    print("\n【どちらが先か】")
    show("ai_first_share", "同じ箇所で AI が先だった割合")

    print("\n【分量 — 人間が読まされる量】")
    per = {}
    for cat in ["ai", "human"]:
        n = df[f"{cat}_comments"].sum()
        vis = df[f"{cat}_chars_visible"].sum()
        per[cat] = vis / max(n, 1)
        print(f"  {cat:<6} コメント {int(n):>9,} 件  平均 {per[cat]:.0f} 文字/件"
              f" (HTMLコメント込みなら {df[f'{cat}_chars'].sum() / max(n, 1):.0f})")
    if per["human"] > 0:
        print(f"  → AI コメント1件は人間コメント **{per['ai'] / per['human']:.1f}件分**の読む量に相当する")

    # 対照との直接比較(同じ PR の中で両方が計算できるものだけ)
    sub = df.dropna(subset=["line_overlap", "human_human_overlap"])
    summary = {
        "n_prs": int(len(df)),
        "tol_lines": TOL,
        "means": {c: float(df[c].dropna().mean()) for c in
                  ["file_overlap", "line_overlap", "human_human_overlap",
                   "ai_covered_by_human", "human_covered_by_ai", "ai_first_share"]
                  if not df[c].dropna().empty},
    }
    if len(sub) >= 30:
        from scipy import stats
        diff = sub["line_overlap"] - sub["human_human_overlap"]
        t, p = stats.ttest_rel(sub["line_overlap"], sub["human_human_overlap"])
        summary["paired_vs_human_human"] = {
            "n": int(len(sub)), "mean_diff": float(diff.mean()),
            "t": float(t), "p": float(p),
            "ai_human": float(sub["line_overlap"].mean()),
            "human_human": float(sub["human_human_overlap"].mean()),
        }
        print(f"\n【対照との比較】同じ PR 内で両方計算できた {len(sub):,} 件")
        print(f"  AI×人間 {sub['line_overlap'].mean():.3f} vs 人間×人間 {sub['human_human_overlap'].mean():.3f}"
              f"  差 {diff.mean():+.3f} (対応あり t検定 p={p:.4g})")
        if p < 0.05 and diff.mean() > 0:
            print("  → AI は人間レビュアーどうしより高い頻度で同じ箇所を指摘している(冗長寄り)")
        elif p < 0.05:
            print("  → AI は人間レビュアーどうしより違う箇所を指摘している(補完寄り)")
        else:
            print("  → 人間レビュアーどうしの重なりと差がない(AI は人間の1人として振る舞っている)")

    (OUT_DIR / "d21_comment_overlap.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'd21_comment_overlap.json'}")


if __name__ == "__main__":
    main()

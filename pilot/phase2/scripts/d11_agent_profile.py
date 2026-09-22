"""D11: エージェント別のレビュー行動プロファイル。

D10 で「AI レビューの 99.6% は COMMENTED、APPROVED は 0.4%、CHANGES_REQUESTED は 6件」
という結果が出た。これが特定のツール(GitHub Copilot のレビュー機能など)の仕様によるものか、
エージェント一般に当てはまるのかを確認する。

「AI には判断権限が与えられていない」が現在の AI レビューの構造だとすれば、
人間が代替されないのは当然の帰結であり、論文の中心的な議論になる。

実行: uv run python pilot/phase2/scripts/d11_agent_profile.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, REPO_ROOT, load_repos, repo_dir_name

# RAW_DIR は common 側で環境変数 PILOT_RAW により切り替わる


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


def main() -> None:
    # login -> {state: count}, login -> category, login -> set(repos)
    by_login: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cat_of_login: dict[str, str] = {}
    repos_of_login: dict[str, set[str]] = defaultdict(set)

    for repo in load_repos():
        actors_path = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        prs_path = RAW_DIR / repo_dir_name(repo) / "prs_full.jsonl"
        if not (actors_path.exists() and prs_path.exists()):
            continue
        actors = pd.read_parquet(actors_path)
        cat_map = dict(zip(actors["actor_login"], actors["category"]))
        for pr in read_jsonl(prs_path):
            for rv in (pr.get("reviews") or {}).get("nodes", []):
                author = rv.get("author")
                if not author:
                    continue
                login = author["login"]
                cat = cat_map.get(login, "unknown")
                if cat not in ("agent", "legacy_bot"):
                    continue
                cat_of_login[login] = cat
                by_login[login][rv.get("state") or "UNKNOWN"] += 1
                repos_of_login[login].add(repo)

    rows = []
    for login, states in by_login.items():
        total = sum(states.values())
        rows.append({
            "login": login,
            "category": cat_of_login[login],
            "n_reviews": total,
            "n_repos": len(repos_of_login[login]),
            "APPROVED": states.get("APPROVED", 0),
            "CHANGES_REQUESTED": states.get("CHANGES_REQUESTED", 0),
            "COMMENTED": states.get("COMMENTED", 0),
            "DISMISSED": states.get("DISMISSED", 0),
            "approved_share": states.get("APPROVED", 0) / total,
            "judgment_share": (states.get("APPROVED", 0) + states.get("CHANGES_REQUESTED", 0)) / total,
        })
    df = pd.DataFrame(rows).sort_values("n_reviews", ascending=False)
    df.to_csv(OUT_DIR / "d11_agent_profile.csv", index=False)

    agents = df[df["category"] == "agent"]
    print("=== agent と判定されたアカウントのレビュー行動(件数上位20) ===")
    print(f"{'login':<42}{'reviews':>9}{'repos':>7}{'APPR':>8}{'CR':>7}{'COMM':>9}{'判断率':>9}")
    print("-" * 91)
    for _, r in agents.head(20).iterrows():
        print(f"{r['login']:<42}{r['n_reviews']:>9,}{r['n_repos']:>7}{r['APPROVED']:>8,}"
              f"{r['CHANGES_REQUESTED']:>7,}{r['COMMENTED']:>9,}{r['judgment_share']:>8.1%}")

    print(f"\nagent アカウント数: {len(agents)}  総レビュー数: {agents['n_reviews'].sum():,}")
    print(f"うち判断(APPROVED + CHANGES_REQUESTED): {agents['APPROVED'].sum() + agents['CHANGES_REQUESTED'].sum():,} "
          f"({(agents['APPROVED'].sum() + agents['CHANGES_REQUESTED'].sum()) / agents['n_reviews'].sum():.2%})")

    # 判断を出しているエージェントが存在するか
    judges = agents[(agents["judgment_share"] > 0.05) & (agents["n_reviews"] >= 20)]
    print(f"\n判断を5%以上出している agent(20件以上レビューしたもの): {len(judges)} 件")
    if len(judges):
        print(judges[["login", "n_reviews", "APPROVED", "CHANGES_REQUESTED", "judgment_share"]].to_string(index=False))

    # 参考: legacy_bot の行動(比較用)
    legacy = df[df["category"] == "legacy_bot"]
    if len(legacy):
        print(f"\n参考) legacy_bot: {len(legacy)} アカウント, 総レビュー {legacy['n_reviews'].sum():,}, "
              f"判断率 {(legacy['APPROVED'].sum() + legacy['CHANGES_REQUESTED'].sum()) / legacy['n_reviews'].sum():.1%}")

    summary = {
        "n_agent_accounts": int(len(agents)),
        "agent_total_reviews": int(agents["n_reviews"].sum()),
        "agent_judgment_share": float(
            (agents["APPROVED"].sum() + agents["CHANGES_REQUESTED"].sum()) / agents["n_reviews"].sum()
        ),
        "top_agents": agents.head(20).to_dict(orient="records"),
    }
    (OUT_DIR / "d11_agent_profile.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd11_agent_profile.csv'}")


if __name__ == "__main__":
    main()

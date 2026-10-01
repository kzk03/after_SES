"""D32: 実行の委譲(マージキュー導入)は人間のレビュー行動を変えるか。

D4 は「AI レビューの導入」を処置として、人間の評価活動が減らないことを示した。
そこでの解釈は「判断が委譲されないから人間は減らない」である。

ならば**実際に委譲が起きた場合**に何が起きるかを見れば、その解釈を直接検定できる。
マージの実行は現に機械へ委譲されている(D30: 人間作成 PR の 0.97%、
cockroachdb では 92%)。しかも導入は階段状のイベントとして観測できる。

  委譲が人間の作業を減らすなら → マージキュー導入後に人間のレビューが減るはず
  委譲されたのが実行だけで判断でないなら → 何も変わらないはず

処置の定義は D4 と揃える。リポジトリ $r$ について、人間作成 PR のうち
機械がマージした割合が「直後3ヶ月の平均が直前3ヶ月の平均より5ポイント以上高く、
かつ20%以上に達する」最初の月を導入時点とする。
(AI レビューより閾値を高くするのは、マージキューは部分導入が長く続くため)

出力: outputs_full/d32_merge_delegation_event.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d32_merge_delegation_event.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (INTERIM_DIR, OUT_DIR, RAW_DIR, build_count_panel, fit_panel,
                    load_repos, repo_dir_name)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "pilot"))
from loop1_actors import classify, load_config  # noqa: E402

JUMP = 0.05      # 直前3ヶ月からの上昇幅
LEVEL = 0.20     # 到達水準
WINDOW = 3
LEAD_LAG = 3


def month_to_idx(s: str) -> int:
    y, m = str(s).split("-")[:2]
    return int(y) * 12 + int(m)


def load_machine_merge_share() -> pd.DataFrame:
    """リポジトリ×月の「人間作成 PR のうち機械がマージした割合」を作る。"""
    cfg = load_config()
    agent_accounts = {a.lower() for a in cfg["agent"]["accounts"]}
    legacy_accounts = {a.lower() for a in cfg["legacy_bot"]["accounts"]}
    legacy_patterns = [re.compile(p) for p in cfg["legacy_bot"].get("patterns", [])]

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
                if mb:
                    by_rule = classify(mb, agent_accounts, legacy_accounts, legacy_patterns)
                    cat = by_rule if by_rule != "human" else cat_of.get(mb, "unknown")
                else:
                    cat = "none"
                rows.append({"repo": repo, "pr_number": d["number"], "merger_cat": cat})
    mb = pd.DataFrame(rows)

    pr = pd.read_parquet(OUT_DIR / "d24_approval.parquet")[
        ["repo", "pr_number", "month", "author_cat"]]
    df = mb.merge(pr, on=["repo", "pr_number"], how="inner")
    df = df[df["author_cat"] == "human"]
    df["is_machine"] = df["merger_cat"].isin(["legacy_bot", "agent"]).astype(float)
    g = df.groupby(["repo", "month"]).agg(n_merged=("is_machine", "size"),
                                          machine_share=("is_machine", "mean")).reset_index()
    return g


def detect_adoption(share: pd.DataFrame) -> dict[str, str]:
    """D4 と同じ形で導入月を検出する。"""
    adoption = {}
    for repo, g in share.groupby("repo"):
        g = g.sort_values("month")
        g = g[g["n_merged"] >= 20]          # 少数月のノイズを除く
        if len(g) < 2 * WINDOW + 1:
            continue
        v = g["machine_share"].to_numpy()
        months = g["month"].tolist()
        for i in range(WINDOW, len(v) - WINDOW + 1):
            pre = v[max(0, i - WINDOW):i].mean()
            post = v[i:i + WINDOW].mean()
            if post - pre >= JUMP and post >= LEVEL:
                adoption[repo] = months[i]
                break
    return adoption


def event_study(panel: pd.DataFrame, adoption: dict[str, str], outcomes: list[str]) -> dict:
    """D4 と同じく PanelOLS で module + month 固定効果を吸収する。

    statsmodels の式で C(module_id_global) を書くとダミーが1万本立って現実的でない。
    """
    p = panel.copy()
    p["adopt_t"] = p["repo"].map(lambda r: month_to_idx(adoption[r]) if r in adoption else np.nan)
    p["t"] = p["month"].map(month_to_idx)
    p["event_time"] = p["t"] - p["adopt_t"]
    treated = p["adopt_t"].notna()

    cols = []
    for k in range(-LEAD_LAG, LEAD_LAG + 1):
        if k == -1:
            continue
        name = f"k_{'m' if k < 0 else 'p'}{abs(k)}"
        if k == LEAD_LAG:
            val = treated & (p["event_time"] >= k)
        elif k == -LEAD_LAG:
            val = treated & (p["event_time"] <= k)
        else:
            val = treated & (p["event_time"] == k)
        p[name] = val.fillna(False).astype(float)
        cols.append(name)

    out = {}
    for y in outcomes:
        try:
            out[y] = fit_panel(p, y, cols)
        except Exception as e:
            out[y] = {"error": str(e)}
    return out


def main() -> None:
    share = load_machine_merge_share()
    adoption = detect_adoption(share)

    print(f"=== 機械マージ比率の系列: {share['repo'].nunique()} リポジトリ ===")
    print(f"検出された「実行の委譲」導入: {len(adoption)} リポジトリ\n")
    for r, m in sorted(adoption.items(), key=lambda kv: kv[1]):
        g = share[(share["repo"] == r)].sort_values("month")
        pre = g[g["month"] < m]["machine_share"].tail(3).mean()
        post = g[g["month"] >= m]["machine_share"].head(3).mean()
        peak = g["machine_share"].max()
        print(f"  {m}  {r:42s} 前 {pre:5.1%} -> 後 {post:5.1%} (最大 {peak:5.1%})")

    never = [r for r in share["repo"].unique() if r not in adoption]
    print(f"\n対照(never-treated): {len(never)} リポジトリ")

    out = {"spec": {"jump": JUMP, "level": LEVEL, "window": WINDOW, "lead_lag": LEAD_LAG},
           "adoption": adoption, "n_never_treated": len(never)}

    if len(adoption) < 2:
        print("\n処置が2件未満のためイベントスタディは実施しない。")
        (OUT_DIR / "d32_merge_delegation_event.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2))
        return

    panel = build_count_panel()
    # 被説明変数は D4 と同じ作り方に揃える
    panel["log_n_E_human"] = np.log1p(panel["n_E_human"])
    panel["log_n_E_agent"] = np.log1p(panel["n_E_agent"])
    panel["log_n_E_human_actors"] = np.log1p(panel["n_E_human_actors"])
    panel["log_n_G_human"] = np.log1p(panel["n_G_human_strict"])
    panel["e_human_per_pr"] = np.where(panel["n_E_prs"] > 0,
                                       panel["n_E_human"] / panel["n_E_prs"], np.nan)
    outcomes = ["log_n_E_human", "log_n_E_human_actors", "e_human_per_pr",
                "H_E", "log_n_E_agent", "log_n_G_human"]
    print(f"\nパネル {len(panel):,} 行 / 被説明変数 {outcomes}")
    res = event_study(panel, adoption, outcomes)
    out["models"] = res

    print("\n=== イベントスタディ: 処置 = 実行の委譲(マージキュー導入)===")
    for y, m in res.items():
        if "error" in m:
            print(f"  {y}: 失敗 {m['error'][:60]}")
            continue
        row = " | ".join(f"{k}: {m['params'][k]:+.4f} (p={m['p_repo'][k]:.3f})"
                         for k in ["k_m3", "k_m2", "k_p0", "k_p1", "k_p2", "k_p3"]
                         if k in m["params"])
        print(f"  {y:22s} n={m['n_obs']:>7,} repos={m.get('n_repos','?')}\n      {row}")

    (OUT_DIR / "d32_merge_delegation_event.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n-> {OUT_DIR / 'd32_merge_delegation_event.json'}")


if __name__ == "__main__":
    main()

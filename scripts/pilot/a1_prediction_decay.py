"""A-1: 予測モデルの環境変化(AI浸透)による劣化を検出する。

`docs/今後の研究方向性.md` のA-1のアイデア(「AI導入前に学習した予測モデルは導入後に性能が落ちる」)を、
元のSES研究(IRL/RFによるレビュー継続予測)の発想のまま、今回のGitHubデータで簡易再現する。
Gerrit専用の元モデルはそのまま移植できないため、ここではシンプルなロジスティック回帰で代用する。

タスク: 開発者がそのモジュールで月tに活動していたら、月t+1にも活動を続けるか(continuation, 二値)。

前半期間(学習)→後半期間(テスト)の時系列分割でAUCを比較する。
  モデルA: 経験・直近活動量のみを特徴量とする(AI関与度を知らない素朴なモデル)
  モデルB: モデルAの特徴量 + モジュールのAI関与度A_prime(t) を追加

モデルAの後半期間でのAUC低下が大きく、モデルBでその低下が緩和されるなら、
「AI浸透という環境変化を知らない予測モデルは劣化するが、AI関与度を明示的に入れれば対応できる」
という主張の裏付けになる。

実行: uv run python scripts/pilot/a1_prediction_decay.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = REPO_ROOT / "pilot" / "interim"
PROCESSED_DIR = REPO_ROOT / "pilot" / "processed"
CONFIG_DIR = REPO_ROOT / "pilot" / "config"

FIRST_BATCH_REPOS = [
    "home-assistant/core",
    "grafana/grafana",
    "juspay/hyperswitch",
    "ray-project/ray",
    "python/cpython",
    "ghostty-org/ghostty",
    "apache/airflow",
    "godotengine/godot",
]


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def load_repos() -> list[str]:
    repos = list(FIRST_BATCH_REPOS)
    batch2_csv = CONFIG_DIR / "loop7_selected_repos_batch2_final.csv"
    if batch2_csv.exists():
        with batch2_csv.open() as f:
            repos += [row["full_name"] for row in csv.DictReader(f)]
    return repos


def build_dataset_for_repo(repo: str) -> pd.DataFrame | None:
    interim_d = INTERIM_DIR / repo_dir_name(repo)
    processed_d = PROCESSED_DIR / repo_dir_name(repo)
    events_path = interim_d / "events.parquet"
    panel_path = processed_d / "panel.parquet"
    if not events_path.exists() or not panel_path.exists():
        return None

    events = pd.read_parquet(events_path)
    ge = events[events["stage"].isin(["G", "E"]) & (events["actor_category"] == "human")]
    if ge.empty:
        return None

    panel = pd.read_parquet(panel_path)[["module_id", "month", "A_prime"]]
    a_lookup = dict(zip(zip(panel["module_id"], panel["month"]), panel["A_prime"]))

    activity = ge.groupby(["actor_login", "module_id", "month"]).size().rename("n_events").reset_index()
    months = sorted(activity["month"].unique())
    month_idx = {m: i for i, m in enumerate(months)}
    if len(months) < 8:
        return None  # 前半/後半に分けるのに短すぎる

    # (actor, module) -> 活動した月インデックスのリスト
    am_months: dict[tuple[str, str], list[int]] = {}
    for a, m, mo in zip(activity["actor_login"], activity["module_id"], activity["month"]):
        am_months.setdefault((a, m), []).append(month_idx[mo])

    rows = []
    for (actor, module), grp in activity.groupby(["actor_login", "module_id"]):
        mon_idx_list = sorted(am_months[(actor, module)])
        mon_set = set(mon_idx_list)
        for t_idx in mon_idx_list:
            if t_idx + 1 >= len(months):
                continue  # 最終月は翌月が観測できない
            t = months[t_idx]
            n_months_active_so_far = sum(1 for x in mon_idx_list if x <= t_idx)
            recent_n_events = grp[grp["month"] == t]["n_events"].sum()
            a_t = a_lookup.get((module, t))
            continues = 1 if (t_idx + 1) in mon_set else 0
            rows.append(
                {
                    "t_idx": t_idx,
                    "n_months_active_so_far": n_months_active_so_far,
                    "recent_n_events": recent_n_events,
                    "A_prime": a_t,
                    "continues": continues,
                }
            )
    if not rows:
        return None
    df = pd.DataFrame(rows).dropna(subset=["A_prime"])
    df["repo"] = repo
    df["n_months_total"] = len(months)
    return df


def main() -> None:
    repos = load_repos()
    frames = []
    for repo in repos:
        df = build_dataset_for_repo(repo)
        if df is not None:
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "a1_prediction_decay_dataset.parquet"
    all_df.to_parquet(out_path, index=False)

    # 各リポジトリのt_idxを0-1にリスケール(月数がリポジトリごとに違うため)し、
    # 全体で前半(train, rel<0.6)・後半(test, rel>=0.6)に分ける
    all_df["t_rel"] = all_df["t_idx"] / (all_df["n_months_total"] - 1)
    train = all_df[all_df["t_rel"] < 0.6].copy()
    test = all_df[all_df["t_rel"] >= 0.6].copy()

    train["log_recent"] = np.log1p(train["recent_n_events"])
    test["log_recent"] = np.log1p(test["recent_n_events"])

    feats_a = ["n_months_active_so_far", "log_recent"]
    feats_b = feats_a + ["A_prime"]

    def fit_eval(feats: list[str]) -> dict:
        X_train, y_train = train[feats], train["continues"]
        X_test, y_test = test[feats], test["continues"]
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        auc_train = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        auc_test = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        return {
            "features": feats,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "auc_train(前半, in-sample)": float(auc_train),
            "auc_test(後半, out-of-sample)": float(auc_test),
            "auc_decay": float(auc_train - auc_test),
            "coef": dict(zip(feats, clf.coef_[0].tolist())),
        }

    model_a = fit_eval(feats_a)
    model_b = fit_eval(feats_b)

    summary = {
        "n_repos_used": len(frames),
        "n_obs_total": len(all_df),
        "n_train": len(train),
        "n_test": len(test),
        "continues_rate_train": float(train["continues"].mean()),
        "continues_rate_test": float(test["continues"].mean()),
        "modelA(経験+直近活動量のみ、AI関与度を知らない)": model_a,
        "modelB(モデルA + A_prime)": model_b,
    }
    (PROCESSED_DIR / "a1_prediction_decay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()

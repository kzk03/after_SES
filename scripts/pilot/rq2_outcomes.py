"""RQ2 — 離脱率・新規参入数の実測(計測。予測ではない)。

研究設計書§5.3:
  離脱率(m,t+k)   ~ 分離度(m,t) + controls + FE
  新規参入数(m,t+k) ~ 分離度(m,t) + controls + FE

従属変数の定義:
  離脱率(m,t,k): 月tにmで活動していた人間担い手のうち、t+1〜t+kに一度も活動しなかった人の割合
  新規参入数(m,t,k): t+1〜t+kに初めてmに現れた人間担い手の数
    (「初めて」はデータ収集期間の開始月以降での初出現。全履歴は見えないための近似)

分離度(m,t)はgates.md Loop6で確定した主軸 S1_EG_prime(= H_E - H_G_prime, trailer込み)を使う。
G/Eイベント(2工程決定, gates.md Loop3)のうち actor_category=human のみを人間担い手として数える。

実行: uv run python scripts/pilot/rq2_outcomes.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

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

K_VALUES = [3, 6]  # 13ヶ月分のデータではk=12だとt=1点しか取れないため3,6を主に使う


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def load_repos() -> list[str]:
    repos = list(FIRST_BATCH_REPOS)
    batch2_csv = CONFIG_DIR / "loop7_selected_repos_batch2_final.csv"
    if batch2_csv.exists():
        with batch2_csv.open() as f:
            repos += [row["full_name"] for row in csv.DictReader(f)]
    return repos


def compute_outcomes_for_repo(repo: str) -> pd.DataFrame | None:
    interim_d = INTERIM_DIR / repo_dir_name(repo)
    processed_d = PROCESSED_DIR / repo_dir_name(repo)
    events_path = interim_d / "events.parquet"
    sep_path = processed_d / "separation_axes.parquet"
    if not events_path.exists() or not sep_path.exists():
        return None

    events = pd.read_parquet(events_path)
    ge = events[events["stage"].isin(["G", "E"]) & (events["actor_category"] == "human")]
    if ge.empty:
        return None

    months = sorted(ge["month"].unique())
    month_idx = {m: i for i, m in enumerate(months)}

    # (module_id, month) -> set of human actor logins
    active: dict[tuple[str, str], set[str]] = {}
    for (mod, mon), grp in ge.groupby(["module_id", "month"]):
        active[(mod, mon)] = set(grp["actor_login"])

    # モジュールごとの「これまでに一度でも現れた人間担い手」の累積集合(新規参入判定用)
    modules = ge["module_id"].unique()

    rows = []
    for mod in modules:
        seen_so_far: set[str] = set()
        for t in months:
            cur = active.get((mod, t), set())
            for k in K_VALUES:
                t_idx = month_idx[t]
                future_months = [months[i] for i in range(t_idx + 1, min(t_idx + 1 + k, len(months)))]
                if len(future_months) < k:
                    continue  # 観測期間の終端でtが足りない場合はスキップ
                future_active: set[str] = set()
                for fm in future_months:
                    future_active |= active.get((mod, fm), set())

                if cur:
                    churned = cur - future_active
                    churn_rate = len(churned) / len(cur)
                else:
                    churn_rate = None

                new_entrants = future_active - seen_so_far - cur
                n_new = len(new_entrants)

                rows.append(
                    {
                        "module_id": mod,
                        "month": t,
                        "k": k,
                        "churn_rate": churn_rate,
                        "n_new_entrants": n_new,
                        "n_active_t": len(cur),
                    }
                )
            seen_so_far |= cur

    outcomes = pd.DataFrame(rows)
    if outcomes.empty:
        return None

    sep = pd.read_parquet(sep_path)[["module_id", "month", "S1_EG_prime", "S1_EG"]]
    merged = outcomes.merge(sep, on=["module_id", "month"], how="left")
    merged["repo"] = repo
    return merged


def main() -> None:
    repos = load_repos()
    frames = []
    skipped = []
    for repo in repos:
        df = compute_outcomes_for_repo(repo)
        if df is None:
            skipped.append(repo)
            continue
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "rq2_outcomes.parquet"
    all_df.to_parquet(out_path, index=False)

    summary = {}
    for k in K_VALUES:
        sub = all_df[all_df["k"] == k]
        valid = sub.dropna(subset=["churn_rate", "S1_EG_prime"])
        summary[f"k={k}"] = {
            "n_obs": len(sub),
            "n_obs_valid(churn_rate & S1 both非null)": len(valid),
            "churn_rate_mean": float(sub["churn_rate"].mean(skipna=True)),
            "n_new_entrants_mean": float(sub["n_new_entrants"].mean()),
            "corr(S1_EG_prime, churn_rate)": float(valid["S1_EG_prime"].corr(valid["churn_rate"])) if len(valid) > 2 else None,
        }
    summary["n_repos_used"] = len(frames)
    summary["n_repos_skipped"] = len(skipped)
    summary["repos_skipped"] = skipped

    (PROCESSED_DIR / "rq2_outcomes_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()

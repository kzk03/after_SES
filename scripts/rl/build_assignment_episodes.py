#!/usr/bin/env python3
"""柱2 のレビュアー割り当てエピソードを生データから生成する。

これまで柱2 が未実行だったのは、env は存在するのに入力のエピソードを作る
生成器が無かったため。本スクリプトがその欠落を埋める。

1行 = 1件のレビュー割り当て実績 (change_id をレビュアー email に振った結果、
label が承諾/非承諾) とみなし、時系列順のリクエスト列に変換する。

生成物:
  - history   : エピソード開始より前の全イベント (各開発者の初期履歴を作る種)
  - episodes  : 時間窓ごとのリクエスト列。各リクエストは
                タスク側特徴 (規模)、実際に振られたレビュアー (logged_reviewer)、
                その結果 (logged_label)、候補集合を持つ

候補集合の作り方:
  リクエスト時点から遡って active_window_days 以内に活動した開発者を
  直近活動順に n_candidates 人まで取る。**実際に振られたレビュアーは必ず含める**
  (含めないとログ方策を再現できず、off-policy 評価もベースライン比較もできない)。

使い方:
    uv run python scripts/rl/build_assignment_episodes.py \
        --raw-csv data/combined_raw_main32.csv \
        --project openstack/nova \
        --start 2021-01-01 --end 2022-12-01 \
        --output outputs/rl/assignment/nova_2021_2022.pkl
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# activity_df の行形式 (extract_common_features が要求する列に合わせる)
ACTIVITY_COLUMNS = [
    "email",
    "timestamp",
    "label",
    "owner_email",
    "change_insertions",
    "change_deletions",
    "change_files_count",
    "first_response_time",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="レビュアー割り当てエピソードの生成")
    p.add_argument("--raw-csv", required=True, help="combined_raw_main32.csv 形式の生データ")
    p.add_argument("--project", default="openstack/nova", help="対象プロジェクト (all で全体)")
    p.add_argument("--start", required=True, help="エピソード期間の開始 (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="エピソード期間の終了 (YYYY-MM-DD, 排他)")
    p.add_argument("--output", required=True, help="出力 pickle パス")
    p.add_argument(
        "--episode-days", type=int, default=30,
        help="1エピソードの長さ (日)。この窓ごとにリクエスト列を切る",
    )
    p.add_argument(
        "--n-candidates", type=int, default=20,
        help="1リクエストあたりの候補者数 (固定長。obs の形状を決める)",
    )
    p.add_argument(
        "--active-window-days", type=int, default=90,
        help="候補とみなす「直近活動あり」の窓 (日)",
    )
    p.add_argument(
        "--min-requests", type=int, default=10,
        help="この件数未満のエピソードは捨てる",
    )
    p.add_argument(
        "--history-days", type=int, default=730,
        help="初期履歴として持ち込む過去の長さ (日)。0 で全期間",
    )
    return p.parse_args()


def load_raw(path: str, project: str) -> pd.DataFrame:
    """生 CSV を読み、activity 行形式に整える。"""
    df = pd.read_csv(path)
    if project and project != "all":
        df = df[df["project"] == project]
        if df.empty:
            raise SystemExit(f"project={project} の行がありません")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "email"])
    df["label"] = df["label"].fillna(0).astype(int)
    for c in ("change_insertions", "change_deletions", "change_files_count"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0).astype(int)
    df["owner_email"] = df["owner_email"].fillna("")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    logger.info(
        "生データ: %d 行, %d 開発者, %s ~ %s",
        len(df), df["email"].nunique(),
        df["timestamp"].min(), df["timestamp"].max(),
    )
    return df


def to_activity_frame(df: pd.DataFrame) -> pd.DataFrame:
    """activity_df 形式 (extract_common_features 用) に射影する。"""
    out = pd.DataFrame({
        "email": df["email"].astype(str),
        "timestamp": df["timestamp"],
        "label": df["label"].astype(int),
        "owner_email": df["owner_email"].astype(str),
        "change_insertions": df["change_insertions"].astype(int),
        "change_deletions": df["change_deletions"].astype(int),
        "change_files_count": df["change_files_count"].astype(int),
        "first_response_time": df.get("first_response_time"),
    })
    return out[ACTIVITY_COLUMNS]


def build_candidates(
    events: pd.DataFrame,
    request_ts: pd.Timestamp,
    logged_reviewer: str,
    n_candidates: int,
    active_window_days: int,
) -> List[str]:
    """リクエスト時点で「直近活動あり」の開発者を候補として返す。

    直近活動順に n_candidates 人。logged_reviewer は必ず先頭以外のどこかに含める
    (ログ方策の再現と off-policy 評価のため)。
    """
    lo = request_ts - pd.Timedelta(days=active_window_days)
    # request_ts 未満に限る (同時刻の自分自身を候補判定に使わない)
    win = events[(events["timestamp"] >= lo) & (events["timestamp"] < request_ts)]
    if win.empty:
        return [logged_reviewer] if logged_reviewer else []

    # 直近活動時刻の降順で並べる
    last_seen = win.groupby("email")["timestamp"].max().sort_values(ascending=False)
    cands = list(last_seen.index[:n_candidates])

    if logged_reviewer and logged_reviewer not in cands:
        # 末尾を1つ落として logged を入れる (候補数を固定に保つ)
        if len(cands) >= n_candidates:
            cands[-1] = logged_reviewer
        else:
            cands.append(logged_reviewer)
    return cands


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    raw = load_raw(args.raw_csv, args.project)
    events = to_activity_frame(raw)

    # ── 初期履歴 (エピソード開始より前) ────────────────────────
    hist_lo = (
        start - pd.Timedelta(days=args.history_days)
        if args.history_days > 0
        else events["timestamp"].min()
    )
    history = events[
        (events["timestamp"] >= hist_lo) & (events["timestamp"] < start)
    ].reset_index(drop=True)
    logger.info(
        "初期履歴: %d 行 (%s ~ %s), %d 開発者",
        len(history), hist_lo.date(), start.date(), history["email"].nunique(),
    )
    if history.empty:
        raise SystemExit("初期履歴が空です。--history-days か --start を見直してください")

    # ── リクエスト列 ──────────────────────────────────────────
    window = raw[(raw["timestamp"] >= start) & (raw["timestamp"] < end)].reset_index(drop=True)
    logger.info("対象期間のリクエスト: %d 件", len(window))
    if window.empty:
        raise SystemExit("対象期間にリクエストがありません")

    # 候補判定には「その時点までに観測済みの全イベント」を使う
    observed = pd.concat([history, to_activity_frame(window)], ignore_index=True)
    observed = observed.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    episodes: List[Dict[str, Any]] = []
    cur: List[Dict[str, Any]] = []
    ep_start = start
    n_skipped = 0

    for row in window.itertuples(index=False):
        ts = row.timestamp
        # エピソード境界を跨いだら切る
        while ts >= ep_start + pd.Timedelta(days=args.episode_days):
            if len(cur) >= args.min_requests:
                episodes.append({
                    "episode_id": len(episodes),
                    "start": ep_start,
                    "end": ep_start + pd.Timedelta(days=args.episode_days),
                    "requests": cur,
                })
            elif cur:
                n_skipped += 1
            cur = []
            ep_start = ep_start + pd.Timedelta(days=args.episode_days)

        cands = build_candidates(
            observed, ts, str(row.email), args.n_candidates, args.active_window_days
        )
        if len(cands) < 2:
            continue  # 選択の余地が無いリクエストは学習に寄与しないので落とす

        cur.append({
            "change_id": str(row.change_id),
            "timestamp": ts,
            "owner_email": str(row.owner_email),
            "change_insertions": int(row.change_insertions),
            "change_deletions": int(row.change_deletions),
            "change_files_count": int(row.change_files_count),
            "logged_reviewer": str(row.email),
            "logged_label": int(row.label),
            "candidates": cands,
        })

    if len(cur) >= args.min_requests:
        episodes.append({
            "episode_id": len(episodes),
            "start": ep_start,
            "end": ep_start + pd.Timedelta(days=args.episode_days),
            "requests": cur,
        })

    n_req = sum(len(e["requests"]) for e in episodes)
    cand_sizes = [len(r["candidates"]) for e in episodes for r in e["requests"]]
    logger.info(
        "エピソード %d 本, リクエスト %d 件 (件数不足で捨てた窓 %d), "
        "候補数 平均%.1f 中央%.0f 最小%d 最大%d",
        len(episodes), n_req, n_skipped,
        float(np.mean(cand_sizes)), float(np.median(cand_sizes)),
        int(np.min(cand_sizes)), int(np.max(cand_sizes)),
    )
    logged_accept = float(
        np.mean([r["logged_label"] for e in episodes for r in e["requests"]])
    )
    logger.info("ログ方策の承諾率: %.4f", logged_accept)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(
            {
                "meta": {
                    "project": args.project,
                    "start": str(start),
                    "end": str(end),
                    "episode_days": args.episode_days,
                    "n_candidates": args.n_candidates,
                    "active_window_days": args.active_window_days,
                    "n_episodes": len(episodes),
                    "n_requests": n_req,
                    "logged_accept_rate": logged_accept,
                },
                "history": history,
                "episodes": episodes,
            },
            f,
        )
    logger.info("保存: %s", out)


if __name__ == "__main__":
    main()

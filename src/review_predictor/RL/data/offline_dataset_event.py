"""オフライン RL データセット (event 単位版)

`offline_dataset.py` (月単位 MDP) とは独立した並行実装。
月単位実装は一切変更せず、目的別にファイルを分離している。

MDP 定義 (event 単位):
  s_t   : event t (1件のレビュー依頼) の到着時点 (timestamp) での dev 特徴量
  a_t   : event t 自体の承諾有無 (Discrete(2))
          月単位版と異なり、集計 (_monthly_accept 相当) は不要。
          生データの `accepted` フラグをそのまま使う。
  r_t   : event t が属する月 m の step_labels[m] + λ · R_θ[m]
          （IRL モデルは月次系列で学習済みのため、報酬は月単位のまま
          その月に属する全 event へ broadcast する）
          ※ transition の "r_theta" 列は評価ベースライン用に
          **前月末の R_θ[m-1]**（イベント到着より過去）を格納する。
          月末の R_θ[m] はイベント自身の結果を入力に含むため、評価に使うと
          IRL 側に事後情報がリークする。シェイピング値は "r_theta_next" 列。
  s_{t+1}: 時系列で次の event (月をまたいでもよい) の特徴量

設計判断:
  - IRL (R_θ) の再学習はしない。月次系列のまま forward_all_steps を呼び、
    月ごとの R_θ を計算してから、各 event にその月の値を配る。
  - action は月次版の `_monthly_accept` のような区間フィルタが不要。
    個々の event dict が直接 accepted/label を持つため、それをそのまま使う。
  - state 計算時の activity_df は「その event の timestamp 以前のイベントのみ」
    に絞る (同月内の未来イベントを含めない) ことで、月単位版で発生した
    累積履歴由来の未来情報混入を event 単位でも確実に防ぐ。
    (event 自身も extract_common_features の `timestamp < feature_end`
    フィルタで除外される = 行動を状態から読み取れない)
  - 特徴量は IRL 学習時 (mce_irl_predictor._extract_one) と同じ定義
    (全履歴窓 + step_total_project_reviews) で再構築する。
    凍結 R_θ に学習時と同じ分布の入力を渡すため。
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from review_predictor.RL.data.offline_dataset import (
    accepted_timestamps_ns,
    dev_accept_label,
)

logger = logging.getLogger(__name__)


def _find_containing_month_index(
    event_ts: pd.Timestamp,
    step_feature_ends: Sequence[Optional[pd.Timestamp]],
) -> Optional[int]:
    """event_ts が属する月インデックス t を返す。

    区間 (step_feature_ends[t-1], step_feature_ends[t]] に event_ts が
    収まる t を探す。見つからなければ None。

    途中に feature_end が解決できない (None の) 月が挟まっている場合、
    その月の本来の境界が分からないため、直前の有効な境界から次の有効な
    境界までの区間は「どちらの月に属すか判定不能」として None を返す
    (次の有効な月に誤って丸ごと帰属させない)。
    """
    prev: Optional[pd.Timestamp] = None
    ambiguous = False
    for t, fe in enumerate(step_feature_ends):
        if fe is None:
            ambiguous = True
            continue
        if not ambiguous and (prev is None or event_ts > prev) and event_ts <= fe:
            return t
        ambiguous = False
        prev = fe
    return None


def _row_from_event(email: str, act: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """1件の生イベント dict を activity_df の行形式に変換する。

    build_transitions (月次版) の行変換ロジックと同一。
    """
    ts = act.get("timestamp")
    if ts is None:
        return None
    return {
        "email": email if act.get("action_type") != "authored" else act.get("reviewer_email", ""),
        "timestamp": pd.Timestamp(ts),
        "label": 1 if act.get("accepted", False) else 0,
        "owner_email": act.get("owner_email", email),
        "change_insertions": act.get("lines_added", act.get("change_insertions", 0)) or 0,
        "change_deletions": act.get("lines_deleted", act.get("change_deletions", 0)) or 0,
        "change_files_count": act.get("files_changed", act.get("change_files_count", 0)) or 0,
        "first_response_time": act.get("response_time", act.get("first_response_time")),
    }


def build_event_transitions(
    trajectories: List[Dict[str, Any]],
    reward_fn,
    state_builder,
    feature_window_days: int = 180,
    reward_shaping: float = 0.3,
    event_action_types: Tuple[str, ...] = ("review",),
    dev_label_window_months: Tuple[int, int] = (0, 3),
    done_mode: str = "episode_end",
    dropout_gap_months: int = 3,
) -> List[Dict[str, Any]]:
    """軌跡リストから event 単位の (s, a, r, s', done) タプルリストを構築する。

    Args:
        trajectories: extract_mce_trajectories.py の出力形式の軌跡リスト
                      (月次版と同じ入力フォーマットをそのまま使う)
        reward_fn: IRLRewardFn インスタンス
        state_builder: RLStateBuilder インスタンス
        feature_window_days: 特徴量抽出窓（日数）
        reward_shaping: R_θ のシェイピング係数 λ
        event_action_types: 行動判定の対象とする event の action_type
                             ("authored" は承諾/非承諾の意思決定が無いため除外)
        dev_label_window_months: step_label_dev (dev 単位継続ラベル) の窓
            (開始月オフセット, 終了月オフセット)。step_labels の抽出時
            future window (0-3 / 3-6 / ...) と揃えること。
        done_mode: エピソード終端の扱い。
            - "episode_end" (既定): 最終 transition を done=1 (従来動作)
            - "dropout": 最終レビューイベントから観測ホライズンまでの
              空白が dropout_gap_months 以上あるときだけ done=1
              (真の離脱)。空白が短ければデータ打ち切りとみなし done=0 で
              bootstrap する
        dropout_gap_months: done_mode="dropout" の離脱判定ギャップ (月)

    Returns:
        transition dict のリスト。月次版と同じキー構成に加えて
        "month_idx" (所属月インデックス) を持つ。
    """
    transitions: List[Dict[str, Any]] = []
    skipped_traj = 0
    skipped_events = 0

    for traj in trajectories:
        developer = traj.get("developer", traj.get("developer_info", {}))
        email = developer.get(
            "email", developer.get("developer_id", developer.get("reviewer", ""))
        )
        directory = traj.get("directory", "")
        monthly_histories = traj.get("monthly_activity_histories", [])
        step_labels = traj.get("step_labels", [])
        step_context_dates = traj.get("step_context_dates", [])
        step_total_project_reviews = traj.get("step_total_project_reviews", [])
        path_features_per_step = traj.get("path_features_per_step", [])
        profile_features_static = traj.get("profile_features", None)

        min_len = min(len(monthly_histories), len(step_labels))
        if min_len < 2 or not monthly_histories:
            skipped_traj += 1
            continue

        # ── 1. 月次の state/action ベクトルを構築し、R_θ を月単位で計算する ──
        # (IRL モデルは月次系列で学習済みのため、報酬計算は月単位のまま変更しない)
        state_vecs: List[np.ndarray] = []
        action_vecs: List[np.ndarray] = []
        step_feature_ends: List[Optional[pd.Timestamp]] = []

        for i in range(min_len):
            month_history = monthly_histories[i]
            feature_end_ts = (
                step_context_dates[i] if i < len(step_context_dates) else None
            )
            if feature_end_ts is None:
                if month_history:
                    # month_history はソート保証がないため、末尾要素ではなく
                    # 最大 timestamp を採用する (未ソート CSV 由来でも正しい)
                    feature_end_ts = max(
                        (
                            pd.Timestamp(a["timestamp"])
                            for a in month_history
                            if a.get("timestamp") is not None
                        ),
                        default=None,
                    )
                if feature_end_ts is None:
                    state_vecs.append(np.zeros(state_builder.state_dim, dtype=np.float32))
                    action_vecs.append(np.zeros(state_builder.action_dim, dtype=np.float32))
                    step_feature_ends.append(None)
                    continue

            feature_end = pd.Timestamp(feature_end_ts)

            rows = [r for r in (_row_from_event(email, act) for act in month_history) if r is not None]
            if not rows:
                state_vecs.append(np.zeros(state_builder.state_dim, dtype=np.float32))
                action_vecs.append(np.zeros(state_builder.action_dim, dtype=np.float32))
                step_feature_ends.append(feature_end)
                continue

            activity_df = pd.DataFrame(rows)
            path_vec = path_features_per_step[i] if i < len(path_features_per_step) else None
            # IRL 学習時 (_extract_one) と同じく step_total_project_reviews を渡す
            total_proj = (
                int(step_total_project_reviews[i])
                if i < len(step_total_project_reviews)
                else 0
            )

            sv = state_builder.build_obs(
                email=email,
                activity_df=activity_df,
                feature_end=feature_end.to_pydatetime(),
                episode_assign_count=0,
                path_features_vec=path_vec,
                profile_features_vec=profile_features_static,
                total_project_reviews=total_proj,
                feature_window_days=feature_window_days,
            )
            av = state_builder.build_action_features(
                email=email,
                activity_df=activity_df,
                feature_end=feature_end.to_pydatetime(),
                total_project_reviews=total_proj,
                feature_window_days=feature_window_days,
            )
            state_vecs.append(sv)
            action_vecs.append(av)
            step_feature_ends.append(feature_end)

        if len(state_vecs) < 2:
            skipped_traj += 1
            continue

        irl_states = torch.tensor(
            np.stack([sv[: state_builder.state_dim] for sv in state_vecs]),
            dtype=torch.float32,
        ).unsqueeze(0)
        irl_actions = torch.tensor(
            np.stack(action_vecs), dtype=torch.float32
        ).unsqueeze(0)
        length = torch.tensor([len(state_vecs)], dtype=torch.long)

        try:
            reward_fn.system.network.eval()
            with torch.no_grad():
                logits_all = reward_fn.system.network.forward_all_steps(
                    irl_states.to(reward_fn.device),
                    irl_actions.to(reward_fn.device),
                    length.to(reward_fn.device),
                )
            logits_all = logits_all[0]
            rewards_all = (logits_all[:, 1] - logits_all[:, 0]).cpu().numpy()
        except Exception as e:
            logger.warning("R_θ 計算エラー (email=%s): %s", email, e)
            skipped_traj += 1
            continue

        # ── 2. 累積履歴の最終スナップショットを master pool として個々の event を抽出 ──
        # monthly_activity_histories[-1] は「開始点から最終月末までの累積履歴」なので
        # 軌跡内の全イベントを含む最も完全なリストになる。
        master_events = monthly_histories[-1]
        review_events = [
            act
            for act in master_events
            if act.get("timestamp") is not None
            and act.get("action_type", "review") in event_action_types
        ]
        review_events.sort(key=lambda a: pd.Timestamp(a["timestamp"]))

        # master pool 全体を 1 回だけ行形式に変換し、timestamp 昇順に整列しておく。
        # 各 event の「その時点までの履歴」は searchsorted による先頭スライスで得る
        # (イベントごとに全リストを再スキャンして DataFrame を作り直す O(E^2) を回避)。
        master_rows = [
            row
            for row in (_row_from_event(email, act) for act in master_events)
            if row is not None
        ]
        master_rows.sort(key=lambda r: r["timestamp"])
        if not master_rows:
            skipped_traj += 1
            continue
        master_df = pd.DataFrame(master_rows)
        master_ts_ns = np.array(
            [r["timestamp"].value for r in master_rows], dtype=np.int64
        )

        # ── 3. 各 event ごとに state を構築 (未来イベントを含めないようスライス) ──
        event_records: List[Dict[str, Any]] = []
        for ev in review_events:
            ts = pd.Timestamp(ev["timestamp"])
            month_idx = _find_containing_month_index(ts, step_feature_ends)
            if month_idx is None:
                skipped_events += 1
                continue

            # timestamp <= ts の行数 (未来イベントを除いた履歴の先頭スライス幅)
            cutoff = int(np.searchsorted(master_ts_ns, ts.value, side="right"))
            if cutoff == 0:
                skipped_events += 1
                continue

            activity_df = master_df.iloc[:cutoff]
            path_vec = (
                path_features_per_step[month_idx]
                if month_idx < len(path_features_per_step)
                else None
            )
            total_proj = (
                int(step_total_project_reviews[month_idx])
                if month_idx < len(step_total_project_reviews)
                else 0
            )
            sv = state_builder.build_obs(
                email=email,
                activity_df=activity_df,
                feature_end=ts.to_pydatetime(),
                episode_assign_count=0,
                path_features_vec=path_vec,
                profile_features_vec=profile_features_static,
                total_project_reviews=total_proj,
                feature_window_days=feature_window_days,
            )
            action_label = 1 if (ev.get("accepted", False) or ev.get("label") == 1) else 0

            event_records.append({
                "timestamp": ts,
                "state": sv,
                "action": action_label,
                "month_idx": month_idx,
            })

        if len(event_records) < 2:
            skipped_traj += 1
            continue

        # dev 単位継続ラベル (step_label_dev) 用: 承諾イベントの昇順 ts 配列と
        # 観測ホライズン (履歴プールが確実にカバーする末尾 = 最後の有効 feature_end)
        acc_ts_ns = accepted_timestamps_ns(master_events)
        horizon = max(
            (fe for fe in step_feature_ends if fe is not None), default=None
        )

        # ── 4. (s_t, a_t, r_t, s_{t+1}, done) を構築 ──
        for t in range(len(event_records) - 1):
            rec = event_records[t]
            m = rec["month_idx"]
            label_m = float(bool(step_labels[m])) if m < len(step_labels) else 0.0
            # R_θ[m] は fe_m (月末) 時点 = イベント到着より後の同月イベント
            # (当該イベント自身の結果を含む) を入力に持つため、報酬シェイピング
            # には使えるが評価ベースラインには使えない (事後情報のリーク)。
            # 評価用の決定時点スコアは前月末の R_θ[m-1] (イベントより厳密に過去)。
            # m=0 のみ前月が無いため R_θ[0] にフォールバックする (初月イベントに
            # 限り同月情報を含む近似。全体の一部でありドキュメント化で許容)。
            r_theta_m = float(rewards_all[m]) if m < len(rewards_all) else 0.0
            r_theta_decision = (
                float(rewards_all[m - 1]) if m >= 1 else r_theta_m
            )
            shaped_reward = label_m + reward_shaping * r_theta_m
            done = t == len(event_records) - 2
            if done and done_mode == "dropout":
                # 最終イベント以降 horizon まで dropout_gap_months 以上の
                # 空白があれば「真の離脱」= terminal。短ければ打ち切り。
                last_ts = event_records[t + 1]["timestamp"]
                done = (
                    horizon is not None
                    and last_ts + pd.DateOffset(months=dropout_gap_months) <= horizon
                )
            # dev 単位継続: step_labels[m] と同じ窓オフセットを fe_m から取る
            fe_m = step_feature_ends[m]
            label_dev = dev_accept_label(
                acc_ts_ns,
                fe_m + pd.DateOffset(months=dev_label_window_months[0]),
                fe_m + pd.DateOffset(months=dev_label_window_months[1]),
                horizon,
            )
            transitions.append({
                "state": rec["state"].tolist(),
                "action": rec["action"],
                "reward": shaped_reward,
                "next_state": event_records[t + 1]["state"].tolist(),
                "done": float(done),
                "email": email,
                "directory": directory,
                "step_label": label_m,
                # step_label_dev: directory を問わない dev 単位継続 (観測不能なら None)
                "step_label_dev": label_dev,
                # r_theta: 決定時点 (イベント到着) より前の月末 R_θ[m-1]。評価ベースライン用
                "r_theta": r_theta_decision,
                # r_theta_next: 報酬シェイピングに使った所属月末の R_θ[m]
                "r_theta_next": r_theta_m,
                "feature_end": rec["timestamp"].isoformat(),
                "month_idx": m,
            })

    logger.info(
        "build_event_transitions 完了: %d transitions, "
        "%d 軌跡をスキップ, %d event をスキップ",
        len(transitions),
        skipped_traj,
        skipped_events,
    )
    return transitions

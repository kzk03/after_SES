"""レビュアー割り当てシミュレータ環境 (柱2)

`review_assignment_env.py` (旧実装) とは独立した並行実装。旧実装は変更せず、
目的別にファイルを分けている (offline_dataset.py / offline_dataset_event.py と同じ方針)。

旧実装からの設計変更 (2026-08-03):

1. **R_θ を報酬から外し、環境の応答モデルに置いた**
   IRL が推定したのは開発者の効用であって、エージェント (推薦システム) の
   目的ではない。開発者は環境の一部なので、「振ったら受けるか」は環境側の
   確率的遷移として扱う。エージェントの報酬はプロジェクト側の指標から設計する。

2. **承諾/拒否を実際にサンプリングする**
   旧実装は割り当てたら即報酬計算で終わり、「拒否」という事象が存在しなかった。
   本実装は AcceptanceModel の確率でベルヌーイ試行し、結果で状態を分岐させる。

3. **結果を開発者の履歴に書き戻す**
   旧実装は activity_df がエピソード中不変で、obs で動くのは割り当て件数の
   1次元だけだった。STATE_FEATURES 26次元のうち 7つ (overall/recent_acceptance_rate,
   accept_rate_drift_30d, acceptance_rate_last10, recent_switch_rate,
   time_since_last_accept, time_since_last_reject) は承諾/拒否履歴に依存するのに
   永久に動かなかった。本実装は結果を履歴に append するため、
   「断られる → 承諾率が下がる → 次の承諾確率が下がる → 選ばれにくくなる」
   というフィードバックが特徴量経由で自然に成立する。

4. **時計を進める**
   旧実装は feature_end 固定で days_since_last_activity 等も動かなかった。
   本実装はリクエストの timestamp をそのまま現在時刻として使う。

5. **負荷を「承諾して抱えている件数」に定義変更**
   旧実装の負荷ペナルティは「振った回数」を使っており、断られた分まで
   負荷に計上していた。本実装は承諾されたものだけを service_days の間
   open review として数える。

6. **候補を順序不変に扱う**
   観測は [n_candidates, feat_dim] の行列として返す (フラット結合しない)。
   候補ごとに共有エンコーダを当てれば並び順に依存しない政策が書ける。
   候補数が足りない場合は action mask で無効化する (旧実装のパディング選択
   ペナルティ -1.0 は不要になる)。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# タスク側特徴の次元 (規模3つ + 対数化した規模3つ + 候補数の正規化)
TASK_FEATURE_NAMES = (
    "log_insertions",
    "log_deletions",
    "log_files",
    "is_large_change",
    "owner_is_candidate",
)


def _gini(x: np.ndarray) -> float:
    """ジニ係数。0 = 完全平等、1 に近いほど一極集中。

    負荷やレビュー件数の偏りを 1 スカラーで測るために使う。
    全員 0 のときは偏りなしとみなして 0 を返す。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    total = x.sum()
    if total <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * xs).sum()) / (n * total) - (n + 1.0) / n)


class _DevState:
    """1開発者のシミュレーション中の状態。

    events はエピソード開始前の実績 + エピソード中にエージェントが生成した
    割り当て結果の両方を含む (activity 行形式の dict のリスト)。
    """

    __slots__ = ("email", "events", "open_until", "n_assigned", "n_accepted", "n_rejected")

    def __init__(self, email: str, events: List[Dict[str, Any]]) -> None:
        self.email = email
        self.events = events
        # 承諾して未完了のレビューの完了予定時刻 (時系列順)
        self.open_until: deque = deque()
        self.n_assigned = 0
        self.n_accepted = 0
        self.n_rejected = 0

    def open_load(self, now: pd.Timestamp) -> int:
        """現在抱えている未完了レビュー件数。期限切れを掃除してから返す。"""
        while self.open_until and self.open_until[0] <= now:
            self.open_until.popleft()
        return len(self.open_until)


class AssignmentSimEnv(gym.Env):
    """レビュアー割り当てシミュレータ。

    観測 (Dict):
        candidates : [n_candidates, feat_dim] 候補ごとの特徴量
        task       : [task_dim] 当該リクエスト (PR) の特徴量
        mask       : [n_candidates] 有効な候補のみ 1

    行動: Discrete(n_candidates) — 候補インデックス
    報酬: プロジェクト側の指標 (承諾・負荷平準化・集中度) の重み付き和
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes: List[Dict[str, Any]],
        history: pd.DataFrame,
        acceptance_model,
        state_builder,
        n_candidates: int = 20,
        service_days: float = 6.0,
        load_cap: int = 5,
        w_accept: float = 1.0,
        w_reject: float = 0.3,
        w_load: float = 0.5,
        w_gini: float = 2.0,
        w_retention: float = 0.0,
        load_clip: Optional[float] = None,
        retention_fn: Optional[Callable[[str, pd.Timestamp], float]] = None,
        feature_window_days: int = 180,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            episodes: build_assignment_episodes.py が出力した episodes
            history: 実ログの全イベント (activity 行形式 DataFrame)。期間全体を
                渡してよい。reset がエピソード開始時刻で切って初期状態にする
            acceptance_model: AcceptanceModel — P(accept | dev state)
            state_builder: RLStateBuilder — 特徴量抽出
            n_candidates: 固定候補数 (観測形状を決める)
            service_days: 承諾したレビューを抱える日数 (実データの応答遅延中央値)
            load_cap: この件数を超える未完了レビューから負荷ペナルティが立つ
            w_accept: 承諾されたときの報酬
            w_reject: 拒否されたときのペナルティ (再依頼コスト)
            w_load: 負荷ペナルティの係数
            w_gini: レビュー集中度 (ジニ係数) のペナルティ係数。
                バス係数の代理指標として使う
            w_retention: 継続スコアの係数。0 なら継続性の項を使わない (既定)
            load_clip: 負荷超過の頭打ち。None ならクリップしない (既定)。
                旧実装の 2.0 相当を再現したいときだけ指定する
            retention_fn: (email, 時点) を受けて継続スコアを返す関数。None なら 0。
                候補プール全員の平均ではなく「選んだ人」の値を使う。プール平均だと
                母集団もキーもエピソード中固定なので定数になり勾配が消えるため
            feature_window_days: 特徴量抽出窓
            seed: 乱数シード
        """
        super().__init__()
        self.episodes = episodes
        self.acceptance_model = acceptance_model
        self.state_builder = state_builder
        self.n_candidates = n_candidates
        self.service_days = service_days
        self.load_cap = max(1, load_cap)
        self.w_accept = w_accept
        self.w_reject = w_reject
        self.w_load = w_load
        self.w_gini = w_gini
        self.w_retention = w_retention
        self.load_clip = load_clip
        self.retention_fn = retention_fn
        self.feature_window_days = feature_window_days

        self.feat_dim = int(state_builder.state_dim)
        self.task_dim = len(TASK_FEATURE_NAMES)

        # 実ログを開発者ごとに時刻順で持っておく。reset では「そのエピソードの
        # 開始時刻より前」の部分だけを切り出して初期状態にする。
        #
        # 全エピソードを同じ初期状態 (期間全体の開始時点) から始めてはいけない。
        # 2022 年のエピソードでも開発者の履歴が 2021 年 1 月で止まることになり、
        # 「直近30日の承諾率」「最後の活動からの経過日数」といった特徴量が
        # 学習時とまったく違う値になる (実測で候補の 51% が直近承諾率 0 になり、
        # 承諾モデルの予測平均が 0.61 → 0.19 に落ちていた)。
        self._base_events: Dict[str, List[Dict[str, Any]]] = {}
        self._base_ts: Dict[str, np.ndarray] = {}
        for email, grp in history.sort_values("timestamp").groupby("email"):
            key = str(email)
            self._base_events[key] = grp.to_dict("records")
            self._base_ts[key] = grp["timestamp"].to_numpy()

        self.observation_space = gym.spaces.Dict({
            "candidates": gym.spaces.Box(
                low=-10.0, high=10.0,
                shape=(n_candidates, self.feat_dim), dtype=np.float32,
            ),
            "task": gym.spaces.Box(
                low=-10.0, high=10.0, shape=(self.task_dim,), dtype=np.float32
            ),
            "mask": gym.spaces.MultiBinary(n_candidates),
        })
        self.action_space = gym.spaces.Discrete(n_candidates)

        self._rng = np.random.default_rng(seed)

        # エピソード状態
        self._devs: Dict[str, _DevState] = {}
        self._requests: List[Dict[str, Any]] = []
        self._ptr = 0
        self._now: Optional[pd.Timestamp] = None
        self._ep_start: Optional[pd.Timestamp] = None
        self._prev_gini = 0.0
        self._feat_cache: Dict[Tuple[str, int, int], np.ndarray] = {}
        # 未介入の開発者の特徴量。エピソードを跨いで再利用するので reset で消さない
        self._base_feat_cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._base_cache_cap = 2_000_000
        self._ep_log: List[Dict[str, Any]] = []
        self._total_events = 0
        self._cand_p_accept = np.zeros(n_candidates, dtype=np.float64)
        self._pool: List[str] = []

    # ------------------------------------------------------------------
    # 特徴量
    # ------------------------------------------------------------------

    def _dev(self, email: str) -> _DevState:
        st = self._devs.get(email)
        if st is None:
            # 履歴を持たない開発者 (初出) も候補になりうる。
            # reset で全員ぶん作るので通常ここは空リストになる
            evs = self._base_events.get(email, [])
            if evs and self._ep_start is not None:
                k = int(np.searchsorted(
                    self._base_ts[email], np.datetime64(self._ep_start), side="left"
                ))
                evs = evs[:k]
            st = _DevState(email, list(evs))
            self._devs[email] = st
        return st

    def _dev_features(self, email: str, now: pd.Timestamp) -> np.ndarray:
        """開発者の特徴量ベクトル [feat_dim]。

        特徴量は日単位の粒度しか持たないので、同じ日のうちは再計算しない。
        キャッシュはエージェントが介入したかどうかで 2 段に分ける:

        - 未介入 (n_assigned == 0): 履歴は初期履歴そのもの。エピソードごとに
          切り出し位置は変わるが、全プロジェクトが同じ 30 日グリッドを使うため
          「ある日付」が属するエピソードは一意で、切り出し位置も一意に決まる。
          よって (email, 日付) で**永続**キャッシュしてよい。
          学習は同じエピソード集合を何度も回すため、ここの再利用が効く。
        - 介入済み: 履歴の中身がエピソードごと (承諾サンプリングの乱数ごと) に
          変わるため永続化できない。エピソード内限定のキャッシュに
          (email, イベント数, 日付) で入れる。1エピソード内では履歴は append
          のみなので、イベント数が中身を一意に決める。
        """
        st = self._dev(email)
        day = now.toordinal()
        untouched = st.n_assigned == 0

        if untouched:
            base_key = (email, day)
            cached = self._base_feat_cache.get(base_key)
            if cached is not None:
                return cached
            key = None
        else:
            key = (email, len(st.events), day)
            cached = self._feat_cache.get(key)
            if cached is not None:
                return cached

        if st.events:
            df = pd.DataFrame(st.events)
        else:
            df = pd.DataFrame(columns=["email", "timestamp", "label", "owner_email"])

        vec = self.state_builder.build_obs(
            email=email,
            activity_df=df,
            feature_end=now.to_pydatetime(),
            episode_assign_count=0,
            path_features_vec=None,
            profile_features_vec=None,
            # 全開発者のイベント総数。毎回 sum を取ると O(開発者数) が
            # 候補数ぶん重なるので、append 時に更新する running counter を使う
            total_project_reviews=max(1, self._total_events),
            feature_window_days=self.feature_window_days,
        ).astype(np.float32)

        if untouched:
            if len(self._base_feat_cache) < self._base_cache_cap:
                self._base_feat_cache[(email, day)] = vec
        else:
            self._feat_cache[key] = vec
        return vec

    def _task_features(self, req: Dict[str, Any], cands: Sequence[str]) -> np.ndarray:
        ins = float(req.get("change_insertions", 0))
        dele = float(req.get("change_deletions", 0))
        files = float(req.get("change_files_count", 0))
        return np.array(
            [
                np.log1p(ins) / 10.0,
                np.log1p(dele) / 10.0,
                np.log1p(files) / 5.0,
                1.0 if (ins + dele) > 500 else 0.0,
                1.0 if req.get("owner_email", "") in cands else 0.0,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def _current_candidates(self) -> List[str]:
        if self._ptr >= len(self._requests):
            return []
        return list(self._requests[self._ptr]["candidates"])[: self.n_candidates]

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO が呼ぶ。有効な候補だけ True。"""
        mask = np.zeros(self.n_candidates, dtype=bool)
        mask[: len(self._current_candidates())] = True
        if not mask.any():
            mask[0] = True  # 全無効だと行動を選べないため保険
        return mask

    def _build_obs(self) -> Dict[str, np.ndarray]:
        cand_mat = np.zeros((self.n_candidates, self.feat_dim), dtype=np.float32)
        mask = np.zeros(self.n_candidates, dtype=np.int8)

        if self._ptr >= len(self._requests) or self._now is None:
            return {
                "candidates": cand_mat,
                "task": np.zeros(self.task_dim, dtype=np.float32),
                "mask": mask,
            }

        req = self._requests[self._ptr]
        cands = self._current_candidates()
        for i, email in enumerate(cands):
            cand_mat[i] = self._dev_features(email, self._now)
            mask[i] = 1

        # 承諾確率は候補全員ぶんまとめて 1 回で推論する。
        # sklearn の predict_proba は 1 行でも 20 行でも呼び出し固定費が支配的なので、
        # step 側で選ばれた 1 人だけ推論するより速く、info にも全候補の値を出せる。
        n = len(cands)
        self._cand_p_accept = np.zeros(self.n_candidates, dtype=np.float64)
        if n:
            self._cand_p_accept[:n] = self.acceptance_model.predict_proba(cand_mat[:n])

        return {
            "candidates": cand_mat,
            "task": self._task_features(req, cands),
            "mask": mask,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options and "ep_idx" in options:
            ep_idx = int(options["ep_idx"]) % len(self.episodes)
        else:
            ep_idx = int(self._rng.integers(0, len(self.episodes)))

        ep = self.episodes[ep_idx]
        self._requests = ep["requests"]
        self._ptr = 0
        self._ep_idx = ep_idx
        self._feat_cache = {}
        self._ep_log = []

        # 開発者状態を「このエピソードの開始時刻」に巻き戻す (履歴は複製して汚さない)。
        # 期間全体の開始時点ではなくエピソード開始時点であることが重要。
        self._ep_start = pd.Timestamp(ep["start"])
        cut = np.datetime64(self._ep_start)
        self._devs = {}
        self._total_events = 0
        for email, evs in self._base_events.items():
            k = int(np.searchsorted(self._base_ts[email], cut, side="left"))
            self._devs[email] = _DevState(email, evs[:k].copy())
            self._total_events += k

        # 集中度を測る母集団 = このエピソードで一度でも候補になった開発者。
        # 全開発者 (履歴に出る数百人) で測るとゼロが多すぎてジニ係数が 0.96-0.99 に
        # 飽和し、報酬の勾配として機能しなくなる。「選べたのに選ばなかった人」まで
        # 含めた偏りを見るのが目的なので、候補プールが正しい母集団。
        self._pool = sorted({e for r in self._requests for e in r["candidates"]})

        self._now = pd.Timestamp(self._requests[0]["timestamp"])
        self._prev_gini = self._pool_gini("n_accepted")
        return self._build_obs(), {
            "ep_idx": ep_idx,
            "n_requests": len(self._requests),
            "pool_size": len(self._pool),
        }

    def _pool_counts(self, attr: str) -> np.ndarray:
        """候補プール内の開発者ごとの件数配列。"""
        return np.array(
            [getattr(self._dev(e), attr) for e in self._pool], dtype=np.float64
        )

    def _pool_gini(self, attr: str) -> float:
        return _gini(self._pool_counts(attr))

    def _retention_score(self, email: str, now: pd.Timestamp) -> float:
        """選んだ開発者の継続スコア。retention_fn 未設定か w_retention=0 なら 0。

        候補プール全員の平均にはしない。参照表方式では値のキーが
        (開発者, 年月) で、1 エピソードは 30 日 = ほぼ 1 ヶ月、候補プールも
        エピソード中固定なので、平均は「どの行動を選んでも同じ定数」になり
        勾配が消える。行動が直接動かせるのは「誰を選んだか」だけなので、
        選んだ人の値をそのまま使う。
        """
        if self.retention_fn is None or self.w_retention == 0.0:
            return 0.0
        return float(self.retention_fn(email, now))

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        assert self._now is not None, "reset() を先に呼んでください"

        req = self._requests[self._ptr]
        cands = self._current_candidates()
        now = pd.Timestamp(req["timestamp"])
        self._now = now

        action = int(action)
        if action >= len(cands):
            # mask が効いていれば起きない。起きたら無効行動として記録し先頭に丸める
            logger.debug("無効な行動 %d (候補 %d 人) — 0 に丸める", action, len(cands))
            action = 0
        chosen = cands[action]
        st = self._dev(chosen)

        # ── 1. 承諾判定 (環境の確率的遷移) ────────────────────
        # 承諾確率は _build_obs で候補全員ぶん計算済み。obs を作った時点の
        # 状態に対する確率なので、行動と同じ時点の値になる。
        p_accept = float(self._cand_p_accept[action])
        accepted = bool(self._rng.random() < p_accept)

        # ── 2. 結果を履歴に書き戻す ───────────────────────────
        # ここが旧実装に無かった部分。承諾率・time_since_last_reject などの
        # 特徴量はこの append を通じて初めて動く。
        st.events.append({
            "email": chosen,
            "timestamp": now,
            "label": 1 if accepted else 0,
            "owner_email": req.get("owner_email", ""),
            "change_insertions": req.get("change_insertions", 0),
            "change_deletions": req.get("change_deletions", 0),
            "change_files_count": req.get("change_files_count", 0),
            "first_response_time": None,
        })
        st.n_assigned += 1
        self._total_events += 1

        load_before = st.open_load(now)
        if accepted:
            st.n_accepted += 1
            # 承諾したレビューは service_days の間その人を占有する
            st.open_until.append(now + pd.Timedelta(days=self.service_days))
        else:
            st.n_rejected += 1

        # ── 3. 報酬 (プロジェクト側の指標) ────────────────────
        #
        # 全項を「1ステップあたりの水準」で揃える。差分 (前ステップとの変化量) で
        # 入れた項は 1 エピソードで (終了時 - 開始時) に畳み込まれてしまい、
        # ステップ数に比例して積み上がる承諾報酬に対して無視できる大きさになる。
        r_accept = self.w_accept if accepted else 0.0
        r_reject = -self.w_reject if not accepted else 0.0

        # 負荷: load_cap を超えて抱えている人に更に振ると線形にペナルティ。
        # 旧実装は min(load_excess, 2.0) で頭打ちにしていたが、上限が
        # -w_load*2.0 = -1.0 と w_accept=1.0 に等しく、「承諾さえされれば
        # 何人ぶん抱えさせても差し引き 0 以上」になっていたためクリップを外した。
        load_excess = max(0, load_before - self.load_cap) / self.load_cap
        if self.load_clip is not None:
            load_excess = min(load_excess, self.load_clip)
        r_load = -self.w_load * load_excess

        # 集中度: 候補プール内の承諾件数のジニ係数 (バス係数の代理)。
        # 旧実装は前ステップとの差分を罰していたため、437 ステップのエピソードでも
        # 合計 -w_gini*(gini_終 - gini_初) ≒ -2 にしかならず効かなかった。水準で罰する。
        gini_now = self._pool_gini("n_accepted")
        r_gini = -self.w_gini * gini_now
        self._prev_gini = gini_now

        # 継続性: 継続しそうな人を選ぶことをプロジェクト側 KPI として評価する。
        # ジニと同じ理由で差分ではなく水準で入れる。
        retention = self._retention_score(chosen, now)
        r_retention = self.w_retention * retention

        reward = r_accept + r_reject + r_load + r_gini + r_retention

        self._ep_log.append({
            "step": self._ptr,
            "chosen": chosen,
            "logged_reviewer": req.get("logged_reviewer", ""),
            "matched_logged": chosen == req.get("logged_reviewer", ""),
            "p_accept": p_accept,
            "accepted": int(accepted),
            "load_before": load_before,
            "load_excess": load_excess,
            "gini": gini_now,
            "retention": retention,
            "reward": reward,
        })

        # ── 4. 次のリクエストへ ───────────────────────────────
        self._ptr += 1
        terminated = self._ptr >= len(self._requests)
        if not terminated:
            self._now = pd.Timestamp(self._requests[self._ptr]["timestamp"])

        info = {
            "chosen": chosen,
            "p_accept": p_accept,
            "accepted": accepted,
            "load_before": load_before,
            "gini": gini_now,
            "retention": retention,
            "r_accept": r_accept,
            "r_reject": r_reject,
            "r_load": r_load,
            "r_gini": r_gini,
            "r_retention": r_retention,
        }
        if terminated:
            info["episode_summary"] = self.episode_summary()

        return self._build_obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------

    def candidate_p_accept(self) -> np.ndarray:
        """現ステップの候補ごとの承諾確率 [n_candidates]。

        greedy ベースライン (承諾確率最大の人を選ぶ) が使う。無効な候補は 0。
        """
        return self._cand_p_accept.copy()

    def candidate_loads(self) -> np.ndarray:
        """現ステップの候補ごとの未完了レビュー件数 [n_candidates]。

        負荷最小ベースラインが使う。無効な候補は大きな値にして選ばれないようにする。
        """
        out = np.full(self.n_candidates, 1e9, dtype=np.float64)
        if self._now is None:
            return out
        for i, email in enumerate(self._current_candidates()):
            out[i] = float(self._dev(email).open_load(self._now))
        return out

    def logged_action(self) -> Optional[int]:
        """実際に振られたレビュアーの候補インデックス (ログ方策の再現用)。"""
        if self._ptr >= len(self._requests):
            return None
        logged = self._requests[self._ptr].get("logged_reviewer", "")
        cands = self._current_candidates()
        return cands.index(logged) if logged in cands else None

    def episode_summary(self) -> Dict[str, float]:
        """エピソード全体のプロジェクト側指標。評価はこれで比較する。"""
        if not self._ep_log:
            return {}
        accepted = np.array([r["accepted"] for r in self._ep_log], dtype=np.float64)
        # 集中度系の指標は候補プール (選べた人の集合) を母集団にする
        loads = self._pool_counts("n_accepted")
        assigned = self._pool_counts("n_assigned")
        active = loads[loads > 0]

        # バス係数の代理: 承諾レビューの 50% を担う最小人数
        srt = np.sort(loads)[::-1]
        csum = np.cumsum(srt)
        half = csum[-1] / 2.0 if csum[-1] > 0 else 0.0
        bus = int(np.searchsorted(csum, half) + 1) if half > 0 else 0

        return {
            "n_requests": float(len(self._ep_log)),
            "accept_rate": float(accepted.mean()),
            "n_accepted": float(accepted.sum()),
            "gini_accepted": _gini(loads),
            "gini_assigned": _gini(assigned),
            "bus_factor_50": float(bus),
            "n_active_reviewers": float(len(active)),
            "max_reviewer_share": float(loads.max() / loads.sum()) if loads.sum() > 0 else 0.0,
            "match_logged_rate": float(
                np.mean([r["matched_logged"] for r in self._ep_log])
            ),
            "mean_p_accept": float(np.mean([r["p_accept"] for r in self._ep_log])),
            "mean_retention": float(np.mean([r["retention"] for r in self._ep_log])),
            "total_reward": float(np.sum([r["reward"] for r in self._ep_log])),
        }

    def episode_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._ep_log)

    def render(self) -> None:
        pass

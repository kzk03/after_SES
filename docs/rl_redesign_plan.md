# RL 再設計プラン — 既存スクリプトの変更点と今後の方針

作成日: 2026-07-04
対象: IRL (MCE-IRL + LSTM) → オフライン RL (BC → IQL) 拡張プロジェクト
目的: SANER 投稿に向けて、状態・行動・報酬・特徴量・モデル・評価の各定義を
網羅的に点検し、「どこを・いつ・なぜ変えるか」を1箇所に集約する。

優先度の凡例:

| 優先度 | 意味 |
|--------|------|
| **A** | 実データ初回実行の**前に**直す（直さないと結果の解釈が壊れる/再実行コストが高い） |
| **B** | 初回実行の結果を見てから v2 で入れる（研究の主張を強くする改良） |
| **C** | Phase 2 / 論文のスコープ判断次第（やらない選択も可） |
| 済 | 2026-07-02〜03 に修正済み |

---

## 0. 修正済み事項（前提の確認）

| 項目 | 内容 |
|------|------|
| 済① | 月次版の因果順序: 行動を月 t+1、報酬を `step_labels[t+1]` に同時シフト。状態から行動を読めるリーク・行動と報酬窓の機械的重なりを解消（`offline_dataset.py`、roadmap §7.6） |
| 済② | R_θ 入力の分布整合: feature_start = 履歴最古（全履歴窓）、`step_total_project_reviews` を軌跡から渡す。IRL 学習時 (`mce_irl_predictor._extract_one`) と同一定義に統一（`state_builder.py`、roadmap §7.7） |

event 版は①の問題が構造的に発生しない（`timestamp < feature_end` の strict フィルタで
イベント自身が状態から除外される）ため、**event 版を主軸**とする方針は変更なし。

## 0.5 月次版と event 版の役割分担（継続予測はどちらで担うか）

- **π(a|s) は継続予測器ではない**: event 版の π は「タスクが到着したという
  条件付きの承諾確率」。単体で「n ヶ月以内に貢献があるか」は出せない。
- **継続予測は V(s)/Q(s,a) が担う**: 報酬そのものが directory 単位の
  3ヶ月継続ラベルなので、Q/V は継続ラベルの（割引累積）予測として学習される。
  タスク到着過程は実ログ系列での学習により経験分布として暗黙に周辺化される
  （到着モデルは不要）。
- **論文スタイル（t0 固定・ペア単位）の継続予測は月次版が対応**:
  月末スナップショット状態がそのまま論文タスクの予測時点に相当する。
  event 版はタスク文脈がある場面（承諾予測・割り当て・遷移分析）を担当。

| モデル | 役割 |
|--------|------|
| 月次版 | スナップショット継続予測 = IRL 論文タスクとの公平比較（RQ1） |
| event 版 | タスク条件付き承諾 + 細粒度 V(s)（RQ2 遷移分析・RQ3 割り当て） |

- **IRL 論文との整合方針**: 元論文のタスク仕様（0-3 窓）を再現しようとすると
  ラベル列・評価軸が乱立して破綻する。**「同じタスクの再現」ではなく
  「同じ土俵での比較」に徹する**: 継続予測タスクは新 MDP から1つだけ定義し
  （= `step_labels[t+1]`、行動月の後に始まる3ヶ月窓 ≒ 1〜4 ヶ月先）、
  IRL・RF・LSTM・RL の全手法を同じ情報 (fe_t まで)・同じ正解で採点する。
  - 行動月のギャップは「意思決定の月」として説明でき、バッファ付き予測として
    方法論的にもきれい（境界の重なりが無い）
  - 凍結 IRL は 0-3 窓で学習済みだが窓の 2/3 が重なるほぼ同一タスク。
    論文に1文明記すれば足りる
  - 教師ありベースライン (RF/LSTM) は最初からこのターゲットで学習させる
  - 元論文の数値との直接比較はしない（引用する先行研究であって再現対象ではない）
- **多期間評価（論文の 4 期間テーブル形式）は再現可能**: 行動月 (fe_t, fe_t+1m]
  と重なるのは最初の窓だけなので、評価期間は {1-4m(シフト), 4-6m, 7-9m, 10-12m}
  となり、**2期間目以降は論文と完全に同じ窓**が使える。
  - 実装: 軌跡抽出の `--multitask` で 4 窓ぶんの `step_labels_per_window` が
    1回の抽出で得られる → transitions にラベル4列を足すだけ（再抽出1回で済む）
  - 評価設計は2通り: (i) 窓ごとに報酬を変えた RL を4本訓練（論文と同じ流儀）、
    (ii) **1つの V(s) を4期間すべてのラベルで採点**。V は割引累積で長期を
    集約するため、元論文で IRL/RF が劣化した遠い期間 (7-9m, 10-12m) で
    劣化しにくいはず — RQ1 の主結果候補。(ii) を主、(i) を ablation に推奨
  - なお「1-4ヶ月先を予測し、次に4ヶ月先から連鎖する」明示的な多段予測は
    未来の状態 s_{t+4} が必要になるため不可。その連鎖は Bellman バックアップが
    V(s) の内部で行っていることであり、別途実装する必要はない

---

## 1. 状態 (state) の定義

### 現状

`STATE_FEATURES(26) + PATH(6) + episode_assign_count(1) = 33次元`。
月次版は月末時点、event 版はイベント到着時点のスナップショット。

### 1-1. 【B】タスク強度が RL の状態から見えない（event 版の最重要拡張）

- **現状の整理**: 強度系の特徴自体は存在する（IRL 論文の「活動特徴 5次元」=
  `ACTION_FEATURES`: avg_action_intensity / avg_change_lines / avg_review_size /
  avg_response_time / repeat_collaboration_rate）。ただし:
  1. これらは IRL の「行動特徴量」として **R_θ の報酬計算にだけ**渡されており、
     BC/IQL が見る観測 (`build_obs` = STATE_FEATURES + PATH) には**入っていない**
  2. 内容も「過去の窓内平均」（②修正後は全履歴平均）であり、
     **到着した依頼そのものの重さ**でも「直近で重いのが連続」でもない
- **変更案**（安価な順に3段階）:
  - (a) `ACTION_FEATURES` 5次元を RL の obs に concat（計算済みの値を繋ぐだけ。
    過去の強度・応答速度の履歴が方策・価値関数に見えるようになる）
  - (b) event 版の状態末尾に**到着イベント自体の特徴**を concat
    （`lines_added+deleted` (log1p)、`files_changed`、`is_cross_project`、
    依頼者との過去協業回数など 3〜5 次元）。イベント dict に既にある
    フィールドで賄えるため軌跡の再抽出は不要
  - (c) 「重いタスクが連続」を直接表す特徴: 直近 k 件の平均タスク規模、
    または規模の移動平均Δ（§1-3 のΔ特徴と統合してよい）
  - いずれも obs_dim が変わるので IQL/BC は再訓練。R_θ に渡す部分
    (state_dim まで) は変えないこと（IRL の入力仕様は凍結）。
- **未来情報との整合性（(b) が成立する理由）**: 依頼は「到着 → dev が意思決定」の
  順なので、**当該イベントの規模・依頼者は意思決定時点で観測済みの文脈**であり
  未来情報ではない（contextual MDP の外生文脈）。入れてはいけないのは
  当該イベントの結果 (`accepted` / `response_time`) と未来イベントの情報のみで、
  どちらも現行実装のフィルタで除外済み。Bellman backup の s' に次タスクの
  特徴が入るのも、データが実到着分布からのサンプルなので期待値として
  正しく周辺化される。シミュレーション利用時も、(i) ログ上の評価と
  (ii) what-if リプレイ（実ログの到着系列を固定し割り当てだけ変える）では
  未来のタスク生成は不要。ログ外へのロールアウト（Phase 3）でのみ
  到着プロセスのシミュレータ（経験分布サンプリング等）が必要になる。
  タスク非依存の価値が必要な場面は経験分布上の平均 V̄(dev状態) で復元できる。
  なお「dev が**自発的に**いつ取り組むか」の予測は到着が外生である本設計の
  スコープ外（point process 等の別モデル領域）と明記しておく。
- **注意**: 依頼対象 directory 情報はイベント dict に**入っていない**
  （`train_model.py` の monthly_activities 構築時に落ちている）。
  「そのディレクトリへの親和度」をタスク特徴にするには
  `train_model.py` 側で dirs を event dict に足す必要がある（軌跡再抽出を伴う）。
  v2 では規模系のみ→v3 で dirs 付きに拡張、の2段構えを推奨。
- **対象**: `offline_dataset_event.py`, `state_builder.py`

### 1-2. 【B】プロジェクト状態の特徴が無い

- **問題**: ビジョンにある「プロジェクトの状態」に対応する特徴がほぼ無い
  （`total_project_reviews` は正規化の分母に使うのみ）。
- **変更案**（安価な順）:
  1. リリースサイクル位相（OpenStack は半年周期。IRL 論文の考察 6.1 でも
     リリース前後の活動変動に言及済み → 話が接続する）
  2. 直近30日のプロジェクト全体レビュー流量（キューの混み具合）
  3. プロジェクト全体の直近承諾率（依頼が滞留しているか）
- **対象**: `common_features.py` に追加 or RL 側で独自に concat。
  IRL の入力次元を変えると R_θ が使えなくなるため、
  **RL 状態にのみ concat する**（state_builder で IRL に渡す部分と分離する）こと。

### 1-3. 【B】状態遷移（変化）を表す特徴の明示化

- **問題**: IQL/BC は MLP でステップ間の文脈を持たない。drift 系特徴
  （`accept_rate_drift_30d`, `activity_trend`, `response_time_trend`）はあるが、
  「拒否が**増えてきた**」「応答が**遅くなってきた**」という遷移パターンの
  表現力は弱い。RQ2（離脱前兆の遷移分析）の主役になる部分。
- **変更案**: 二択。
  - (a) **Δ特徴**: 前ステップとの差分（承諾率Δ、応答時間Δ、活動量Δ等）を
    5次元程度 concat。実装が最も軽い
  - (b) **frame-stacking**: 直近 k ステップ（k=3 程度）の状態を連結して
    obs_dim×k にする。系列文脈を持たせる最安の方法で、
    recurrent Q/V の実装（コスト大）を回避できる
- **推奨**: まず (a)、V(s) の Spearman や遷移分析の結果を見て (b) を追加検討。
- **対象**: `offline_dataset*.py`（Δ計算）, `train_offline_rl.py`（obs_dim）

### 1-4. 【A→判断のみ】episode_assign_count は Phase 1 では死んだ次元

- **問題**: 常に 0 のまま学習される（Phase 2 の負荷分散用に予約した次元）。
  害は小さいが、論文で「33次元のうち1つは常時0」は説明に窮する。
- **変更案**: Phase 1 の論文では **obs から外す**（`state_builder.obs_dim` を
  state_dim と等しくするフラグを追加）か、外さないなら論文に書かない。
  コスト最小なので初回実行前にフラグだけ入れておくのが楽。
- **対象**: `state_builder.py`, パイプラインの `OBS_DIM` 計算

### 1-5. 【B】特徴量の棚卸し（見直し・追加の全リスト）

#### 見直し（削除でなく「解釈注意」— R_θ 用は IRL 学習時と整合のため現状維持）

| 特徴 | 問題 | 扱い |
|------|------|------|
| `window_tenure_days` | cap=180日。全履歴窓（修正②）では半年以上の dev 全員が 1.0 飽和 → ほぼ定数 | R_θ 用は維持。RQ2 の解釈には使わない |
| `active_months_ratio` | 分母が窓全月数のため「直近の定期性」→「生涯活動密度」に意味が変化 | 同上 |
| `total_changes`/`total_reviews` | log1p+cap500 で長寿 dev は圧縮 | log なので許容 |
| `episode_assign_count` | Phase 1 では常に 0（§1-4） | obs から除去 |
| `ACTION_FEATURES` 5次元 | R_θ 専用経路で obs 未接続（§1-1(a)） | obs に concat |
| profile 5次元 | 抽出経路で未付与（§1-6） | profile variant 使用時に対応 |

#### 追加候補（仮説→特徴の対応。合計 10 次元前後に絞る）

| クラスタ | 特徴 | 対応する仮説 | コスト |
|----------|------|-------------|--------|
| 拒否・応答悪化 | 直近の連続非承諾 streak | 「拒否が増える→離脱」（drift 系は率のみで「連続」を表せない） | 小 |
| 〃 | 直近30日応答時間の絶対水準（中央値） | `response_time_trend` は傾きのみで水準が無い | 小 |
| 〃 | 直近30日の無応答率 | 明示的拒否と無応答の分離（§2-1 と連動） | 小 |
| タスク強度 | 直近10件の平均変更行数・ファイル数 | 「重いタスクが連続」（既存 avg 系は全履歴平均で薄まる） | 小 |
| 〃 | サイズドリフト（直近10件平均 − 全期間平均） | 「重いのが**増えてきた**」 | 小 |
| 〃 | (event版) 到着タスク自体の規模 | §1-1(b) | 小 |
| 負荷 | 判定時点の未応答依頼数（7日制限なし） | キュー滞留（`concurrent_open_reviews_7d` の一般化） | 小 |
| 〃 | 依頼到着数トレンド（30d/90d 比） | 依頼が増えてきているか | 小 |
| プロジェクト状態 | リリースサイクル位相 | 半年周期の活動変動（IRL 論文 6.1 の考察と接続） | 最小（timestamp のみ） |
| 〃 | プロジェクト依頼流量 | `step_total_project_reviews` の隣接差分で計算可 → **再抽出不要** | 最小 |
| モジュール状態 | directory の活動レビュアー数（bus factor）・依頼数 | 負荷分散・代替可能性 | **要再抽出**（イベントに dirs が無い） |

#### 実装の規律

- R_θ に渡す state_dim までの部分は凍結。**追加分は RL obs の末尾 concat のみ**
- 追加は 10 次元前後に絞る（transition 数に対する過学習回避・RQ2 の解釈性）
- RQ2 用に特徴量を意味クラスタ（負荷/拒否傾向/応答遅延/タスク強度/親和度/
  プロジェクト状態）に分け、**クラスタ単位の group ablation** で
  「どの信号群が V(s) の予測に効くか」を示す
- 再抽出が必要な項目（directory 系・イベント dirs・multitask ラベル §0.5）は
  **1回の再抽出にまとめて**実施する

### 1-6. 【C】profile 特徴の未接続

`extract_mce_trajectories.py` 経由の軌跡には `profile_features` が付与されない
（付与するのは `train_mce_irl.py` の学習パスのみ）。profile 入り variant
（state_dim 37/42）で RL を回す場合は抽出スクリプトに attach 処理を足すこと。
また `build_obs` は path=None かつ profile≠None のとき profile が path の
スロットにずれて入る潜在バグがある（現行 variant では未発火）。
profile variant を使う段階で両方直す。

---

## 2. 行動 (action) の定義

### 現状

月次版: 月 t+1 に承諾レビューを1件以上行ったか（Discrete(2)、修正済み①）。
event 版: そのイベント自体の `accepted`（Discrete(2)）。

### 2-1. 【B】「非承諾」は「明示的拒否」と「無応答」の混合

- **問題**: `accepted=False` には「見たが断った」と「そもそも応答しなかった」が
  混ざる。行動として意味が異なり（拒否は意思決定、無応答は不在の可能性）、
  「拒否が増える → 離脱」という遷移仮説の検証では区別したい。
- **変更案**: `response_time` (first_response_time) の有無で
  3値化できる: {承諾, 応答あり非承諾, 無応答}。
  Discrete(3) は IQL 側は `n_actions` を変えるだけで対応可能な設計になっている。
  ただし eval の `iql_q_advantage = Q[:,1]-Q[:,0]` は **n_actions=2 前提**なので
  多値化時に要修正（例: Q[accept] − max(Q[non-accept系])）。
- **推奨**: 初回は binary で回し、行動再現性（BC の精度）と
  無応答率の実測を見てから 3値化を判断。
- **対象**: `offline_dataset*.py`（行動判定）, `eval_assignment.py`（advantage 計算）

### 2-2. 【B→判断】行動 (dev 単位) と報酬 (directory 単位) の非対称

- **事実**: 行動判定に使う履歴は `reviewer_history`（dev の全 directory・全プロジェクト）、
  一方 `step_labels` は**軌跡の directory に限定した**継続ラベル。
  つまり「他の directory のレビューを承諾した」ことが action=1 になり、
  その報酬は「当該 directory で継続したか」で与えられる。
- **論点**: これはバグではなく設計選択だが、V(s) の意味が
  「この (dev, directory) ペアの残存価値」なのか「この dev の残存価値」なのか
  曖昧になる。§3-1 のラベル選択とセットで決めること。
- **対象**: 設計判断（コード変更は §3-1 に従属）

### 2-3. 【C】行動の多値化・連続化（Phase 1.5）

{非承諾, 承諾のみ, 承諾+author, …} の Discrete(K) 化、あるいは
負荷量を含む連続行動。IQL は n_actions 変更のみで対応可。
初回結果と論文のストーリー（RQ3 をやるか）が決まってから。

---

## 3. ラベル・報酬 (reward) の定義

### 現状

月次版: `r_t = step_labels[t+1] + λ·R_θ[t+1]`（修正済み①）。
event 版: `r_t = step_labels[m] + λ·R_θ[m]`（イベントの所属月 m に broadcast）。
λ 既定 0.3。

### 3-1. 【B・重要な設計判断】directory 単位継続 vs dev 単位継続

- **問題**: 研究のゴールが「開発者の継続確率」なら、報酬は
  dev レベルの継続（どの directory でもよいから活動継続）の方が自然。
  現状の step_labels は directory 限定なので、V(s) は
  「dev をこの directory に留められるか」を表す。
- **変更案**: build 時に **dev 単位の継続ラベルを別列 (`step_label_dev`) として
  併記**する（master_events から「fe 以降3ヶ月に承諾があるか」を計算するだけ。
  軌跡再抽出不要）。初回実行で両方の列を出し、
  V(s) の Spearman・AUC を両ラベルで比較してから主指標を決める。
  IRL (R_θ) は directory 単位で学習済みなので、dev 単位を主報酬にするなら
  λ の意味（directory 信号での shaping）も論文で説明を揃えること。
- **対象**: `offline_dataset*.py`, `eval_assignment.py`

### 3-2. 【B】event 版の報酬 broadcast による return 膨張

- **問題**: 月報酬を月内全イベントにコピーするため、イベントが多い dev ほど
  割引和が大きくなる。V(s) が「継続性」でなく「依頼頻度」に支配される恐れ。
  さらに 1 step = 1 event なので γ の実時間スケールが dev ごとに不均一。
- **変更案**（択一）:
  - (a) 報酬をその月のイベント数で割る（return が月次版と同スケールになる）
  - (b) semi-MDP 的にイベント間隔 Δt（月単位）で `γ^Δt` を使う
    （IQL の `_q_loss` で γ を transition ごとの値にする。実装は小さい）
  - (c) そのままにして V の解釈で注意する（初回はこれで可）
- **推奨**: 初回 (c) → V(s) と依頼頻度の相関を実測し、
  相関が支配的なら (b) を導入（時間的に最も正当）。
- **対象**: `offline_dataset_event.py`, `iql.py`（(b) の場合）

### 3-3. 【B】done=1 が「データ打ち切り」を終端扱いしている

- **問題**: 各軌跡の最終 transition を done=1 にしているが、これは
  エピソード終了ではなく観測期間の終わり（timeout/truncation）。
  V が観測末期の状態で系統的に過小評価される。V(s) が主指標なので影響あり。
- **変更案**: 「真の離脱」と「打ち切り」を区別する:
  - 打ち切り（期間末尾）: done=0 のまま bootstrap（s' は存在するので正しい）
  - 真の終端: 「以降データ末尾まで一切活動がない」場合のみ done=1
    （master_events の最終 timestamp と比較して判定）
- **推奨**: 初回実行はそのままでよい（IRL との比較には影響が対称）。
  v2 で導入し、V(s) の Spearman が改善するか ablation として報告する。
- **対象**: `offline_dataset*.py`

### 3-4. 【A→判断のみ】弱負例の sample_weight が RL に伝搬していない

- **事実**: 軌跡には PU 学習由来の `sample_weight`（依頼なし月の弱負例 = 0.5）が
  あるが、RL の transition には引き継がれず全サンプル等重みで学習している。
- **変更案**: 初回は等重みで可（IRL 論文と扱いが変わる点として認識だけする）。
  入れるなら BC の cross-entropy と IQL の各損失に重み列を掛けるだけ。
  論文の Threats か設計節にどちらを選んだか明記すること。
- **対象**: `offline_dataset*.py`（列追加）, `bc_policy.py`, `iql.py`

### 3-5. 【B】λ (reward_shaping) sweep は計画どおり必須

λ ∈ {0, 0.1, 0.3, 0.5, 1.0}。**λ=0 と 0.3 で差がなければ IRL 依存を外し
「RL 単独」の物語に簡素化する**（論文構成の判断材料なので早めに）。
`REWARD_SHAPING` 環境変数で回せる状態になっている。

---

## 4. モデル設計

### 現状

BC: 3層 MLP + LayerNorm。IQL: 自前実装（V: expectile τ=0.9 / Q: Bellman +
100step ハード同期 / π: advantage-weighted BC β=3.0）、γ=0.99。

### 4-1. 【B】γ=0.99 は月次スケールでは実質無割引

- **問題**: 月次 MDP で γ=0.99 は実効地平 ~100 ヶ月。軌跡長は最大 24 ヶ月なので
  V はほぼ「残り全部の和」になり、割引の意味が薄い。
  （旧ドキュメントの γ=0.95 と実装 0.99 の不整合は roadmap 側を 0.99 に修正済み）
- **変更案**: γ ∈ {0.9, 0.95, 0.99} を hyperparameter sweep に含める。
  「3ヶ月先」を重視する研究文脈なら 0.9〜0.95 が自然。
- **対象**: `run_rl_pipeline*.sh` に `IQL_GAMMA` 環境変数を追加（現状 CLI 引数のみ）

### 4-2. 【B】系列文脈の欠如（§1-3 と同根）

IRL は LSTM で「時系列変動を捉える」のが売りなのに、RL 側の Q/V/π が
スナップショット MLP なのは論文として突っ込まれやすい。
対応は §1-3 の Δ特徴 / frame-stacking を優先し、
recurrent Q/V（実装コスト大・オフラインRLでの学習不安定リスク）は C 扱い。

### 4-3. 【C】細部の改良（効果小、余裕があれば）

- q_target のハード同期 (100 step) → Polyak soft update（安定性）
- BC の val split が transition ランダム分割 → dev 単位分割
  （同一 dev-month の実質重複が train/val を跨ぎ、early stopping が楽観的）
- `hidden_dim`/`batch_size` 等の sweep は主要結果確定後に

### 4-4. 変更不要と判断した点

- BC → IQL の2段構成、IRL 凍結（再学習しない）方針: 維持
- IQL 自前実装の損失定義: レビュー済み・論文 (Kostrikov 2021) と整合
- `sigmoid(R_θ)` を irl_pi とする近似: `softmax([R(s,0),R(s,1)])[1]` と
  数学的に等価なので問題なし

---

## 5. 評価 (eval_assignment.py)

### 5-1. 【A】実データ実行の前に直す3点

| # | 問題 | 修正 |
|---|------|------|
| 1 | Ensemble: test 側の V を **test の min/max** で正規化しており、テスト統計が評価時に混入（train-fit → test 適用の原則違反） | train の min/max を保存して test に適用 |
| 2 | `remaining_accept` が email 単位 groupby で、同一 dev の複数 (directory) 軌跡ブロックを跨いで累積。V(s) の Spearman（主指標）の意味がずれる | transitions に directory 列を追加し (email, directory) 単位で逆累積 |
| 3 | F1: スコア自身の min/max 正規化 + 閾値0.5（保留 #3）。実質 AUC の言い換え | train で Youden/最適F1閾値を fit して test に固定適用 |

※ #2 は build 側に directory 列を足す必要がある（`traj.get("directory")` を
transition dict に入れるだけ）。**初回実行前に入れないと再ビルドになる**ので A。

### 5-2. 【B】ベースラインの拡充（SANER 対策）

- **教師あり LSTM**（同一特徴・同一分割）: 「RL である必要性」への直接回答。最優先
- RF / XGBoost: IRL 論文との接続（RF は前論文のベースライン）
- 生存分析 (Cox): 継続予測の定番手法として査読者が期待する
- dev ホールドアウト split (`--holdout-devs-ratio`): 未知 dev への汎化

### 5-3. 【B】統計的厳密性

- 複数 seed（≥5）+ 既存の bootstrap CI 流儀を踏襲
- 複数 t0（rolling origin）: IRL 論文で自ら threats に書いた「単一 t0」への回答。
  t0 sweep 基盤は既存（`outputs/t0_sweep_align*`）
- Wilcoxon + 効果量（前論文の作法と揃える）

---

## 6. 優先度サマリ

### A: 実データ初回実行の前に（**2026-07-04 実装済み**）

1. 済 — eval 3点修正（§5-1）: Ensemble の V 正規化範囲と F1 閾値を train-fit →
   test 固定適用に変更、remaining_accept を (email, directory) 単位に修正。
   transitions に directory 列を追加
2. 済 — episode_assign_count を obs から除外（`include_assign_count` フラグ、
   ビルド CLI 既定オフ、パイプラインの OBS_DIM = STATE_DIM に変更）
3. 済 — `step_label_dev` 列の併記（§3-1）。窓オフセットは `--dev-label-window`
   (パイプラインが FW_START,FW_END を自動で渡す)。観測ホライズンを超える窓は
   None (観測不能)。eval に `RL_*_step_label_dev` 軸を追加済み
4. 済 — sample_weight は**等重みで開始**と決定（Threats に明記する。
   重み付けは必要になったら §3-4 の手順で導入）

### 初回実行（A の直後）

- event 版 + 月次版、λ ∈ {0, 0.3} の最小 sweep、`is_held_out` / train-test AUC 差 /
  V(s) Spearman（directory 単位 vs dev 単位の両方）を確認
- **この結果で決めること**: IRL を残すか（λ の効き）、主ラベル（§3-1）、
  V(s) の頻度バイアスの深刻度（§3-2）

### B: v2（結果を見て、7月中〜8月）

- event 版タスク特徴（§1-1）→ 中心仮説「重タスク連続」の検証に必須
- Δ特徴 or frame-stacking（§1-3）、プロジェクト状態特徴（§1-2）
- 済 (2026-07-04) — γ sweep（§4-1）: `run_rl_sweep.sh` + `IQL_GAMMA` 環境変数
- 済 (2026-07-04) — done/terminal 再定義（§3-3）: `--done-mode dropout`
  として実装済み（既定は従来動作。ablation として回して比較する）
- γ^Δt or 報酬正規化（§3-2）→ 初回結果で V と依頼頻度の相関を見てから
- 行動3値化の判断（§2-1）→ 初回結果を見てから
- 済 (2026-07-04) — 教師ありベースライン (RF/LogReg) + dev ホールドアウト
  （§5-2）: eval に組み込み済み。系列教師あり (LSTM) は IRL がその役割を
  兼ねるため追加実装しない方針（irl_pi = 時系列教師ありの位置づけで書く）
- 複数 t0 / 複数 seed（§5-3）→ `SEED` は配線済み。t0 は sweep を
  t0_window 引数で回すだけ（軌跡キャッシュが窓ごとに必要）

### C: スコープ判断次第（8月末に決定）

- 行動多値化 Discrete(K)（§2-3）、負荷ペナルティ報酬（Phase 2）
- recurrent Q/V（§4-2）
- RQ3（割り当て方策の what-if リプレイ + OPE）: `train_online_rl.py` /
  `review_assignment_env.py` は行動空間不整合の休眠コードなのでゼロから作り直し

---

## 7. 研究方針（SANER 2027 想定）

- **主張の柱**: RQ1 = V(s)（長期エンゲージメント価値）が教師あり/IRL を超えるか、
  RQ2 = どの状態遷移（拒否増加・重タスク連続・応答遅延・プロジェクト位相）が
  V 低下＝離脱前兆か、RQ3 = Q/V による依頼先選択の what-if（縮小可）
- **因果の書き方に注意**: 行動はログ上、遷移にも報酬にも因果介入しない。
  advantage は「相関」であり「介入効果」と書かない（査読の地雷）
- **スケジュール**: 7月前半 P0 実行と物語確定 → 7月後半 A/B 前半 →
  8月 B 後半 + RQ3 判断 → 9月 執筆 + replication package → 10月頭 投稿
- **フォールバック**: RQ2 中心の分析論文に再構成 / SANER ERA track /
  MSR 2027（1月頃）・ICSME 2027

---

## 8. 実装ログ

### 2026-07-02〜03: 致命傷の修正（済①②）+ A 項目

§0 と §6-A を参照。

### 2026-07-04: 「実行するだけ」段階までの実装

初回実行 + λ/γ 感度 + 未知 dev 汎化 + 教師あり比較 + 結果集約を、
コード変更なしで回せる状態にした。テストは合成軌跡での
end-to-end（build → 分割保存 → train CLI → eval CLI → 集約）まで全 PASS。

#### 新規ファイル

| ファイル | 内容 |
|----------|------|
| `src/review_predictor/RL/data/loading.py` | 共通ローダー（軌跡 / 凍結IRL / transitions の state 行列化 / **報酬の再計算**）。build×2・train・eval に逐語重複していた処理を一元化 |
| `scripts/rl/run_rl_sweep.sh` | λ×γ sweep ランナー。**データセット構築は粒度ごとに1回だけ**行い、λ は train 時に列から再計算（5点振ってもビルド1回） |
| `scripts/rl/summarize_rl_results.py` | 全 run の summary_metrics.json を横断集約 → long CSV + run×手法の markdown 表（軸: step_label / step_label_dev / monthly / V Spearman） |
| `scripts/rl/analyze_value_drivers.py` | **推奨考察手法 (RQ2)**: V(s) の特徴グループ permutation 感度 + ΔV↔Δ特徴の遷移 Spearman + V 急落遷移プロファイル。CSV/図/md を出力 |
| `scripts/rl/compare_methods_stats.py` | 手法間 AUC 差の paired bootstrap（95%CI + p値、IRL 論文 表8 と同じ流儀）。pair_predictions.csv から任意ターゲット・任意 baseline で実行 |
| `docs/rl_experiment_pipeline.md` | **実験→評価→考察→執筆の実行パイプライン**。判断ゲート（λ/ラベル/V/汎化）、考察の方向性 17 項目（推奨・コスト付き）、RQ→図表マッピング、threats チェックリスト |

#### 全コード検証で発見・修正した欠陥 (2026-07-04)

1. **event 版の IRL ベースラインへの事後情報リーク（修正済み）**:
   event transition の `r_theta` が「所属月**末**の R_θ[m]」で、イベント
   （意思決定）より後の同月イベント＝当該イベント自身の結果を入力に含んでいた。
   月次版は①修正で決定時点/シェイピングを分離済みだったが event 版が未対応。
   → `r_theta` = 前月末の R_θ[m-1]（決定時点より過去。m=0 のみ R_θ[0]
   フォールバック）、`r_theta_next` = R_θ[m]（シェイピング用）に分離。
   報酬値・λ 再計算 (`recompute_reward`) は無変更で互換。
2. **全件一括 forward の OOM リスク（修正済み）**: eval の `compute_scores` と
   analyze の V 計算が train 全体を1テンソルで forward していた。event 粒度の
   train は 10^5〜10^6 transition になり得るためバッチ分割 (8192) に変更。

#### 機能追加

| 対象 | 内容 |
|------|------|
| `train_offline_rl.py` | `--reward-shaping λ`（保存済み reward を無視して `step_label + λ·R_θ` を列から再計算 = **λ sweep で再ビルド不要**）、`--seed`（BC split / torch 初期化の再現性）、`train_config.json` 出力（集約用） |
| `eval_assignment.py` | **教師ありベースライン** `rf_supervised` / `logreg_supervised`（train states → step_label で fit、test に適用。「RLである必要性」への直接回答。`--skip-supervised-baselines` で無効化可） |
| `dataset_io.py` | **dev ホールドアウト** `--holdout-devs-ratio`。email の決定的ハッシュで選ぶため月次/event で同じ dev 集合が holdout され、粒度間比較が可能。時間分割より先に除外するので train/test に漏れない |
| 両ビルダー | `--done-mode {episode_end, dropout}`。dropout: 月次=全 done0（打ち切り bootstrap）、event=最終イベント後のギャップ≥`--dropout-gap-months` のときのみ done=1（真の離脱）。既定は従来動作 |
| 両パイプライン | 環境変数 `IQL_GAMMA` / `IQL_TAU` / `IQL_BETA` / `SEED` / `HOLDOUT_DEVS_RATIO` を配線。devholdout ファイルがあれば `eval_devholdout/` に未知 dev 評価を自動出力 |
| 依存関係 | `pyarrow`（parquet 書き出しに必須なのに未宣言だった — 実データ実行で確実に落ちるバグ）と `tabulate`（markdown 表）を pyproject に追加 |

#### リファクタリング

- build スクリプト×2 の軌跡/IRLロード（~30行×2）と train/eval の `_to_vec`
  重複を `loading.py` に集約
- `OfflineRLDataset.from_arrays()` を追加し、train のロードを
  行ごとの dict 生成（iterrows）からベクトル化構築に変更
- eval のサマリ表示3ブロックを手法リスト1つに統一

#### 推奨実行手順（サーバ側）

```bash
# (0) 前提: IRL モデルと軌跡キャッシュ。無ければ従来どおり
#     bash scripts/rl/run_rl_pipeline.sh 0-3 lstm_baseline cpu で Step1 を通す

# (1) 推奨: sweep 一発（event+月次 × λ5点、dev holdout 10% 込み、集約まで自動）
bash scripts/rl/run_rl_sweep.sh 0-3 lstm_baseline cpu

# (2) 単発で回す場合
SKIP_TRAJ_EXTRACT=1 HOLDOUT_DEVS_RATIO=0.1 bash scripts/rl/run_rl_pipeline_event.sh 0-3 lstm_baseline cpu

# (3) 結果の読み方: outputs/rl/sweep/summary.md を見る
#   - RL_iql_pi / iql_v vs irl_pi / rf_supervised の step_label AUC (主仮説)
#   - step_label vs step_label_dev (§3-1 のラベル設計判断)
#   - λ=0 と λ>0 の差 (IRL を残すかの判断)
#   - eval vs eval_devholdout (未知 dev への汎化)
#   - RL_iql_v_long_term の Spearman ≥ 0.3 か
```

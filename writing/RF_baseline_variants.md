# RF ベースライン: 1snap 版 と 月次パネル 版の実装まとめ

**作成日**: 2026-06-02
**最終更新**: 2026-06-03（dynstart 整合版 1snap / panel を追加）
**目的**: 論文の RF ベースラインを「学習データ露出量・特徴量抽出期間を IRL と揃える」設計で複数用意した。新しい結果（IRL DYNAMIC_START 学習、B 定義ラベル等）で再実行する際の手順書。

---

## TL;DR

`outputs/main32_mce_volat_ep150/` の **既存 RF_Dir は 6ヶ月窓 1スナップショット学習**。これに加えて、IRL（3年分の月次軌跡）とデータ量を揃える設定を順次追加：

1. **1snap 3y (固定窓、legacy)**: 同じ 1スナップショット学習だが窓を 3年に拡張（`WINDOW_DAYS=1095`）
2. **月次パネル 3y (固定 train_start、legacy)**: 各月で累積特徴量を計算してパネル化、RF を 1 回学習
3. **dynstart 1snap（**新規・主結果向け**）**: サーバ実験 `dynstartB_ep150` と整合させるため、horizon 別に train_start を前倒し
4. **dynstart 月次パネル（**新規・主結果向け**）**: 同じく horizon 別 train_start で月次パネルを構築し、全 horizon で panel 月数 33 に揃える

すべての変種で：
- **特徴量は IRL と同一** (`FEATURE_NAMES_WITH_PATH` = state 26 + path 6 + action 5 = 37 次元)
- **学習ラベルは B 定義**（1 回でも承諾 → 1）
- **推論候補プールは 180 日窓で固定**（→ 全 horizon で 4,534 ペアに揃う）
- 違いは「学習データの時間軸の使い方」のみ

サーバ IRL (`outputs/dynstartB_ep150/`) との全モデル比較では 3-4 番（dynstart 整合版）を採用する。1-2 番は legacy として残す。

---

## 1. RF ベースラインの一覧

| 種類 | 特徴量計算窓 | 学習サンプル | スクリプト | 出力ディレクトリ | 用途 |
|---|---|---|---|---|---|
| **6m 1snap (既存)** | [TRAIN_END − 180日, TRAIN_END] | ~5,000 (dev, dir) | `eval_mce_irl_path_prediction.py` 内 RF_Dir 部分 | `outputs/main32_mce_volat_ep150/.../pair_predictions.csv` の `rf_dir_prob` | legacy |
| **3y 1snap (固定窓)** | [TRAIN_END − 1095日, TRAIN_END] | ~5,000 (dev, dir) | `scripts/local/run_rf_only_local.sh` (`WINDOW_DAYS=1095`) | `outputs/local_rf_w1095/fs_*/pair_predictions.csv` | legacy |
| **3y monthly panel (固定 train_start)** | 各月 m で [2019-01-01, m] 累積（horizon により 33/30/27/24 月） | ~170,000–311,000 (dev, dir, month) | `scripts/local/run_rf_panel_local.sh` | `outputs/local_rf_panel/fs_*/rf_panel_predictions.csv` | legacy |
| **dynstart 1snap (新)** | [2019-01-01 − FS(h)月, 2022-01-01]（horizon 別） | ~5,000 (dev, dir) | `scripts/local/run_rf_dynstart_1snap_local.sh` | `outputs/dynstart_rf_1snap/fs_*/pair_predictions.csv` | **dynstartB IRL との比較用** |
| **dynstart 月次パネル (新)** | 各月 m で [2019-01-01 − FS(h)月, m] 累積（全 horizon 33 月） | 241,000–311,000 (dev, dir, month) | `scripts/local/run_rf_dynstart_panel_local.sh` | `outputs/dynstart_rf_panel/fs_*/rf_panel_predictions.csv` | **dynstartB IRL との比較用** |

### 1.1 共通の前提

- **学習データ**: `data/combined_raw_main32.csv` (32 main repos)
- **raw_json**: `data/combined_raw_main32.raw_json_list.txt`（32 JSON の半角スペース区切り）
- **TRAIN_START** = 2019-01-01（パネルのみ使用）
- **TRAIN_END** = 2022-01-01（学習ラベル基準）
- **PREDICTION_TIME** = 2023-01-01（推論時点）
- **horizon**: future_start_months ∈ {0, 3, 6, 9}（3 ヶ月幅で 4 horizons）
- **特徴量**: `FEATURE_NAMES_WITH_PATH`（37 次元）
- **モデル**: `RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)`

### 1.2 ラベル定義（重要）

| 段階 | 6m 1snap (volat) | 3y 1snap | 3y monthly |
|---|---|---|---|
| 学習 train ラベル | **B 定義** ([rf_predictor.py:147-157](../src/review_predictor/IRL/model/rf_predictor.py)) | **B 定義** (同上) | **B 定義** ([eval_rf_panel.py:81-94](../scripts/analyze/eval/eval_rf_panel.py)) |
| pair_predictions の `label` 列 | **A 定義** ([eval_mce_irl_path_prediction.py:1229](../scripts/analyze/eval/eval_mce_irl_path_prediction.py)) | **A 定義** (同上) | **B 定義** ([eval_rf_panel.py:275-279](../scripts/analyze/eval/eval_rf_panel.py)) |

**ラベル定義の詳細**: `writing/HANDOFF_label_definition_audit.md` 参照

→ **比較する時は B 定義で揃える** 必要あり（`/tmp/full_compare_aligned.py` のロジック参照）

---

## 2. 1snap 3y 版の実装

### 2.1 思想

「volat の 6m RF と同じ実装をそのまま使い、特徴量窓だけ 3年に伸ばす」。最小限の変更で IRL のデータ露出量 (3年) に揃える。

### 2.2 実行コマンド

```bash
cd /Users/kazuki-h/Master_research
WINDOW_DAYS=1095 OUTBASE=outputs/local_rf_w1095 \
  bash scripts/local/run_rf_only_local.sh
```

### 2.3 内部動作

`scripts/local/run_rf_only_local.sh` は各 horizon ごとに：

```bash
uv run python scripts/analyze/eval/eval_mce_irl_path_prediction.py \
    --data data/combined_raw_main32.csv \
    --raw-json "${RAW_JSON[@]}" \
    --irl-model /dev/null/no_irl_model.pt \   # IRL を強制 skip
    --prediction-time 2023-01-01 \
    --delta-months 3 \
    --future-start-months $fs \
    --rf-future-start-months $fs \
    --rf-train-end 2022-01-01 \
    --window-days 1095 \                      # ★ ここが 6m→3y
    --device cpu \
    --n-jobs 4 \
    --output-dir outputs/local_rf_w1095/fs_${fs}_to_${fs+3}m
```

→ IRL モデルパスが存在しないので IRL_Dir/IRL ブロックは全部 skip → **RF_Dir + Naive + Linear + RF (global) のみ走る**

### 2.4 注意点：候補ペアプールも変わる

`eval_mce_irl_path_prediction.py:696` の：

```python
window_start = prediction_time - pd.Timedelta(days=window_days)
dir_developers = get_directory_developers(df, window_start, prediction_time)
```

これも `window_days` を使うので、**WINDOW_DAYS=1095 にすると推論候補が 3 年分に広がる**（180 日: 4,534 ペア → 1095 日: **18,192 ペア**）。base_rate も 0.27 → 0.084 と低くなる。

→ volat 6m と直接比較するなら、4,534 ペアに絞って再評価する必要あり。`/tmp/full_compare_aligned.py` 参照。

### 2.5 出力

`outputs/local_rf_w1095/fs_<X>_to_<Y>m/`:
- `pair_predictions.csv`: 列構成は volat と同じ（`developer, directory, label, irl_dir_prob, ..., rf_dir_prob, ...`）
  - **`label` は A 定義**
  - `irl_dir_prob` は None（IRL skip しているため）
  - `rf_dir_prob` が 3y 1snap RF の予測値
- `summary_metrics.json`: Naive/Linear/RF/RF_Dir のメトリクス

---

## 3. 月次パネル 3y 版の実装

### 3.1 思想

IRL の月次軌跡（36 ステップ）と同じ情報量を与えるため、**各月 m で累積特徴量を計算して 1 サンプルとして積む**。RF は静的なので時系列構造は学習しないが、「異なる時点での特徴量パターンとラベルの対応」を膨大なサンプルで学習する。

### 3.2 実行コマンド

```bash
cd /Users/kazuki-h/Master_research
bash scripts/local/run_rf_panel_local.sh

# 高速化したい場合 (3ヶ月おき = 12 サンプル/horizon)
MONTH_STEP=3 OUTBASE=outputs/local_rf_panel_step3 \
  bash scripts/local/run_rf_panel_local.sh
```

### 3.3 主要パラメータ（env 上書き可）

| パラメータ | 既定値 | 意味 |
|---|---|---|
| `TRAIN_START` | 2019-01-01 | パネル開始月 |
| `TRAIN_END` | 2022-01-01 | パネル終了月 (label_end ≤ TRAIN_END でクリップ) |
| `PREDICTION_TIME` | 2023-01-01 | 推論時点 |
| `DELTA_MONTHS` | 3 | ラベル窓長 |
| `HORIZONS` | `0 3 6 9` | future_start_months 列 |
| `MONTH_STEP` | 1 | 何ヶ月おきにスナップショットを撮るか |
| `N_JOBS` | 4 | 並列度 |

### 3.4 内部動作

`scripts/analyze/eval/eval_rf_panel.py` の構成：

#### Step 1: 月次パネル構築 (`build_monthly_panel`)

```python
months = pd.date_range(train_start, train_end, freq='MS')[::month_step]
for m_end in months:
    label_start = m_end + DateOffset(months=fs)
    label_end = label_start + DateOffset(months=delta)
    if label_end > train_end: continue          # ラベルリーク防止クリップ

    # joblib parallel で各月を抽出
    extract_one_month(df, train_start, m_end, label_start, label_end, path_extractor)
```

各月の `extract_one_month` で：
- feature window: `[train_start, m_end]`（累積）
- 各 `(dev, dir)` ペアについて B 定義ラベル
- 特徴量: `extract_common_features` + `path_extractor.compute` → 37 次元

#### Step 2: パネルキャッシュ

```python
cache_path = Path("outputs/local_rf_panel/_cache") / f"panel_ts{ts}_te{te}_fs{fs}_dm{dm}_ms{ms}.pkl"
if cache_path.exists():
    panel = pd.read_pickle(cache_path)
else:
    panel = build_monthly_panel(...)
    panel.to_pickle(cache_path)
```

同じ条件（同じ train_start/train_end/fs/delta_months/month_step）の再実行は **キャッシュ即ロード**。

#### Step 3: RF 学習

```python
X_train = panel[FEATURE_NAMES_WITH_PATH].values
y_train = panel["label"].values
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                              random_state=42, n_jobs=-1).fit(X_train, y_train)
```

学習データ量は horizon により異なる：
- 0-3m: 33 月 × ~5K pairs = **311K 行**（46K 正例）
- 3-6m: 30 月 × ~5K pairs = **262K 行**
- 6-9m: 27 月 × ~5K pairs = **215K 行**
- 9-12m: 24 月 × ~5K pairs = **171K 行**

#### Step 4: 推論

```python
feature_start_inf = prediction_time - Timedelta(days=window_days)   # 既定 180 日
# 推論候補: prediction_time 時点で過去 180 日に dev が触った全 dir
# → volat 6m と同じ 4,534 ペアが対象になる
```

**1snap 3y と違って `--window-days` は推論候補プールにのみ影響**（学習は monthly panel で別途完結しているため）。

推論はバッチ化（200 ペア/バッチ）して joblib parallel で実行：

```python
BATCH_SIZE = 200
batches = [pair_only[i:i+BATCH_SIZE] for i in range(0, len(pair_only), BATCH_SIZE)]
# 各バッチ内で email ごとに common_feats をキャッシュ
batch_results = Parallel(n_jobs=4, backend="loky")(
    delayed(_predict_batch)(b) for b in batches
)
```

⚠️ 過去にここで joblib pickle が `/tmp` を満杯にした（ペア単位 delayed だと df を 4,534 回 pickle）。**バッチ化必須**。

### 3.5 出力

`outputs/local_rf_panel/fs_<X>_to_<Y>m/`:
- `rf_panel_predictions.csv`:
  - 列: `email, directory, rf_panel_prob, actual`
  - **`actual` は B 定義**（同期間内に承諾あり → 1）
- `rf_panel_feature_importance.json`: 学習済み RF の Gini importance
- `summary_metrics.json`:
  - `pair_auc_roc`, `pair_auc_pr`, `base_rate`, `n_pairs`
  - `train_panel_size`, `train_unique_devs`, `train_unique_dirs`, `train_unique_months`

`outputs/local_rf_panel/_cache/`: パネル `.pkl`（horizon 切替・再実行高速化用）

---

## 3b. dynstart 整合版（1snap / 月次パネル、2026-06-03 追加）

### 3b.1 思想

サーバの IRL dynstartB 実験は **horizon (FS) ごとに train_start を `FS` 月だけ前倒し、ラベル窓が train_end を超える月をクリップ**することで、月次ステップ数を全 horizon 33 に揃える設計（`ses/README.md` 参照）。RF 側でも IRL と完全等化させるため、両端を horizon 別に shift する設計を採用。

**1snap dynstart**（両端 shift, 学習窓 33 月で完全等化、Option B 採用）:
- horizon 別 `rf_train_start = 2019-01-01 − FS(h) 月`
- horizon 別 `rf_train_end = 2022-01-01 − (FS + delta)(h) 月` = **IRL dynstartB の最終 m_end と同じ**
- 学習窓 = **全 horizon 33 月**で完全等化
- 推論候補プールは **180 日窓で固定**（→ 全 horizon 4,534 ペアで揃う）
- 推論特徴窓は学習窓と同じ長さ（33 月）に自動で揃える
- 訓練ラベル窓 = `[rf_train_end + FS, rf_train_end + FS + delta]` = `[2021-10, 2022-01]`（全 horizon 共通）

**dynstart 月次パネル**（train_start のみ shift, ラベルクリップで自然等化）:
- horizon 別 `train_start = 2019-01-01 − FS(h) 月`、`train_end = 2022-01-01` 固定
- `eval_rf_panel.py` 内のラベルクリップ (`label_end ≤ train_end`) で**全 horizon panel 月数 = 33** に自動等化（IRL と同じロジック）
- 推論候補プール・特徴窓は `eval_rf_panel.py` の既定 180 日

両方とも `outputs/dynstartB_ep150/.../irl_dir_prob` と直接アンサンブル可能。

#### 別案: Option A（train_end 固定）

参考として、`outputs/dynstart_rf_1snap_optA/` に「train_end=2022-01-01 固定・train_start のみ shift」版 (Option A) も保存。学習窓は 36/39/42/45 月で IRL の総期間とは一致するが、データ量が horizon でばらつく。主結果には Option B を使う。

### 3b.2 実行コマンド

```bash
# 1snap dynstart（4 horizon, ~70 分／全 CPU 1 並列）
bash scripts/local/run_rf_dynstart_1snap_local.sh
# → outputs/dynstart_rf_1snap/fs_<X>_to_<Y>m/pair_predictions.csv

# 月次パネル dynstart（4 horizon, ~70-80 分）
bash scripts/local/run_rf_dynstart_panel_local.sh
# → outputs/dynstart_rf_panel/fs_<X>_to_<Y>m/rf_panel_predictions.csv
```

env で上書き可能なパラメータ：

| 変数 | 既定値 | 意味 |
|---|---|---|
| `BASE_TRAIN_START` | 2019-01-01 | FS=0 時の rf_train_start |
| `BASE_TRAIN_END` (1snap) / `TRAIN_END` (panel) | 2022-01-01 | FS=0 時の rf_train_end / panel の固定 train_end |
| `PREDICTION_TIME` | 2023-01-01 | 推論時点 |
| `POOL_WINDOW_DAYS` (1snap のみ) | 180 | 候補プール窓 |
| `HORIZONS` | "0 3 6 9" | future_start_months 列 |
| `DELTA_MONTHS` | 3 | ラベル窓長 |
| `MONTH_STEP` (panel のみ) | 1 | 何ヶ月おきにスナップショット |
| `N_JOBS` | 4 | 並列度 |
| `OUTBASE` | `outputs/dynstart_rf_{1snap,panel}` | 出力先 |

### 3b.3 内部実装

**1snap dynstart**:
- `eval_mce_irl_path_prediction.py` に 2 引数を追加（2026-06-03）：
  - `--rf-train-start YYYY-MM-DD`: RF_Dir 学習窓の開始日。指定時は `--window-days` を無視。推論特徴窓もこの長さに自動で揃える。
  - `--pool-window-days N`: 候補ペアプール窓を学習窓と分離。未指定なら `--window-days` と同値（後方互換）。
- wrap スクリプトが Python で `BASE_TRAIN_START − FS(h)` と `BASE_TRAIN_END − (FS+delta)(h)` を計算し、`--rf-train-start` と `--rf-train-end` の両方を horizon 別に渡す。

**dynstart 月次パネル**:
- `eval_rf_panel.py` は既存の `--train-start` をそのまま使用（コード変更なし）。
- wrap スクリプトが horizon 別に `--train-start` を上書きするだけ。
- panel 側は train_end 固定でラベルクリップにより自然等化（IRL と同じロジック）。
- パネルキャッシュも train_start の違いで別ファイルに自動分離。

### 3b.4 実測結果（2026-06-03 ローカル、Option B）

**1snap dynstart**（Option B, 学習窓 33 月で完全等化）:

| horizon | rf_train_start | rf_train_end | 学習窓長 | RF AUC_ROC | RF AUC_PR | 推論ペア |
|---|---|---|---|---|---|---|
| 0-3m | 2019-01-01 | 2021-10-01 | 1004 日 (33 月) | 0.779 | 0.754 | 4,534 |
| 3-6m | 2018-10-01 | 2021-07-01 | 1004 日 (33 月) | 0.768 | 0.704 | 4,534 |
| 6-9m | 2018-07-01 | 2021-04-01 | 1005 日 (33 月) | 0.768 | 0.661 | 4,534 |
| 9-12m | 2018-04-01 | 2021-01-01 | 1006 日 (33 月) | 0.789 | 0.658 | 4,534 |

（日数の微差は月日数の違い由来、月数は全 horizon 33 で完全等化）

**月次パネル dynstart**:

| horizon | train_start | panel 月数 | panel 行数 | pair_AUC_ROC | pair_AUC_PR | base_rate |
|---|---|---|---|---|---|---|
| 0-3m | 2019-01-01 | 33 | 311,130 | 0.833 | 0.693 | 0.270 |
| 3-6m | 2018-10-01 | 33 | 289,312 | 0.852 | 0.692 | 0.225 |
| 6-9m | 2018-07-01 | 33 | 271,575 | 0.829 | 0.594 | 0.217 |
| 9-12m | 2018-04-01 | 33 | 240,979 | 0.805 | 0.526 | 0.179 |

→ 全 horizon で 4,534 ペア・33 学習月に揃い、IRL dynstartB との公平比較が可能。

**Option A 参考結果**（train_end 固定、学習窓 36/39/42/45 月、`outputs/dynstart_rf_1snap_optA/`）:

| horizon | rf_train_start | 学習窓長 | RF AUC_ROC | RF AUC_PR |
|---|---|---|---|---|
| 0-3m | 2019-01-01 | 1096 日 | 0.792 | 0.774 |
| 3-6m | 2018-10-01 | 1188 日 | 0.810 | 0.750 |
| 6-9m | 2018-07-01 | 1280 日 | 0.774 | 0.681 |
| 9-12m | 2018-04-01 | 1371 日 | 0.792 | 0.658 |

---

## 4. 結果（参考、B 定義で揃えた 4,534 ペアでの AUC_ROC / AUC_PR）

ラベル B 定義 + 推論ペアは volat 6m と同じ 4,534 で揃え。`/tmp/full_compare_aligned.py` で再計算した結果（legacy 比較）：

| horizon | IRL_Dir (volat) | RF 6m (volat) | **Ensemble** | RF 3y 1snap (aligned) | RF 3y monthly (aligned) |
|---|---|---|---|---|---|
| 0-3m | 0.809/0.654 | 0.821/0.705 | **0.842/0.719** | 0.811/0.698 | 0.833/0.693 |
| 3-6m | 0.801/0.597 | 0.857/0.701 | **0.864/0.695** | 0.839/0.672 | 0.847/0.687 |
| 6-9m | 0.787/0.560 | 0.818/0.587 | **0.840/0.615** | 0.793/0.571 | 0.817/0.577 |
| 9-12m | 0.780/0.484 | 0.803/0.532 | **0.815/0.552** | 0.806/0.529 | 0.788/0.488 |

**観察（legacy 表）**:
- RF をどの方式で強化しても **Ensemble を超えない**
- 6m RF と 3y monthly はほぼ同等（±0.02）→「RF の本質はスナップショット特徴の質、月次パネル化は無効果」
- IRL alone は最弱だが、Ensemble 相棒として価値あり

**dynstart 整合版（2026-06-03 追加）**: サーバ IRL dynstartB の `irl_dir_prob` 回収後、§8 の `rf_variant_archetype_gain.py` を `SOURCES` 差し替えで再実行し、同形式の表を再生成する。

---

## 5. 新しい実験での再実行手順

### 5.1 dynstartB IRL（サーバ）との全モデル比較フロー（主推奨、2026-06-03）

```
[ローカル] RF dynstart 整合版を回す（IRL 待ちと並行可）
  bash scripts/local/run_rf_dynstart_1snap_local.sh
      → outputs/dynstart_rf_1snap/fs_*/pair_predictions.csv
  bash scripts/local/run_rf_dynstart_panel_local.sh
      → outputs/dynstart_rf_panel/fs_*/rf_panel_predictions.csv

[サーバ] IRL dynstartB が完了したら rsync で回収
  rsync -avz socsel:/mnt/data1/kazuki-h/Master_research_v2/outputs/dynstartB_ep150/ \
    /Users/kazuki-h/Master_research/outputs/dynstartB_ep150/

[ローカル] §8 のアーキタイプ利得分析を dynstart 構成で再生成
  - rf_variant_archetype_gain.py の SOURCES を以下に差し替え:
      ("dynstart_1snap", "volat_fs", "outputs/dynstart_rf_1snap", "rf_dir_prob")
      ("dynstart_panel", "panel_fs", "outputs/dynstart_rf_panel",  "rf_panel_prob")
  - IRL 確率の base を outputs/dynstartB_ep150/lstm_baseline に切替
  - 実行: uv run python writing/tmp/scripts/rf_variant_archetype_gain.py
  → アンサンブル AUC / archetype 利得 / horizon 別表 が再生成される
```

ポイント：
- **RF は IRL に依存しないので、IRL のサーバ実行完了を待たずローカルで先回し可能**
- **候補ペアプール 4,534 が全モデルで揃う**（1snap は `POOL_WINDOW_DAYS=180`、panel は `eval_rf_panel.py` の既定 180 日推論窓）
- **ラベルは B 定義（承諾あり）に統一**（`rf_variant_archetype_gain.py` が `combined_raw_main32.csv` から再計算）

### 5.2 legacy（固定窓 6m / 3y / 固定 train_start panel）を redo する場合

`scripts/analyze/eval/eval_mce_irl_path_prediction.py` 側の修正が必要：
- pair_predictions の `label` 列を B 定義に変更（[HANDOFF_label_definition_audit.md](./HANDOFF_label_definition_audit.md) 参照）

その上で：

```bash
# 3y 1snap (固定窓)
WINDOW_DAYS=1095 OUTBASE=outputs/local_rf_w1095_v2 \
  bash scripts/local/run_rf_only_local.sh

# 月次パネル (固定 train_start)
PREDICTION_TIME=2024-01-01 \
  TRAIN_END=2023-01-01 \
  OUTBASE=outputs/local_rf_panel_2024 \
  bash scripts/local/run_rf_panel_local.sh
```

**キャッシュ注意**: `outputs/local_rf_panel/_cache/` の `.pkl` は TRAIN_START/TRAIN_END/fs/delta/month_step の組み合わせで命名。条件変えれば自動再構築。dynstart 版は別 `_cache` ディレクトリを使うので衝突なし。

### 5.3 ラベル定義 B で全部統一するなら

`writing/HANDOFF_label_definition_audit.md` の Step 1-2 を先に実行：
1. `pair_predictions.csv` に `label_B` 列を追加（既存ファイル）
2. `pair_trajectories.csv` を B 版で再生成

そのあと downstream の分析を redo。

---

## 6. 既知の落とし穴

### 6.1 joblib のディスク溢れ

過去に発生：4,534 ペアを個別 delayed で並列化したら df を毎回 pickle して `/tmp` が満杯（OSError [Errno 28]）。

**対策**: バッチ化（`eval_rf_panel.py:288-307`）。新規スクリプト書くときは同様にバッチ化する。

### 6.2 候補ペアプールの違い

`--window-days` を変えると **学習だけでなく推論候補プールも変わる** (`eval_mce_irl_path_prediction.py:696`)。fair な比較には推論側も volat と同じプールに揃える必要あり。

### 6.3 月次パネルでの label リーク防止

```python
if label_end > train_end:
    continue   # スキップ
```

future_start_months が大きい horizon ほど使える月が減る（0-3m: 33 月、9-12m: 24 月）。

### 6.4 パネルキャッシュの古さ

`outputs/local_rf_panel/_cache/panel_*.pkl` はファイル名で条件を識別している。**実装を変更した場合（特徴量定義変更、`extract_one_month` の挙動変更等）はキャッシュを削除する**。

```bash
rm -rf outputs/local_rf_panel/_cache/
```

---

## 7. 関連ファイル

### 実装

- `scripts/local/run_rf_only_local.sh` — 1snap 用ラッパー（legacy 固定窓）
- `scripts/local/run_rf_panel_local.sh` — 月次パネル用ラッパー（legacy 固定 train_start）
- `scripts/local/run_rf_dynstart_1snap_local.sh` — **dynstart 1snap 用ラッパー（新規 2026-06-03）**
- `scripts/local/run_rf_dynstart_panel_local.sh` — **dynstart 月次パネル用ラッパー（新規 2026-06-03）**
- `scripts/analyze/eval/eval_rf_panel.py` — 月次パネル本体（panel 全変種で共用）
- `scripts/analyze/eval/eval_mce_irl_path_prediction.py` — 1snap 全変種で利用（2026-06-03 に `--rf-train-start` / `--pool-window-days` 追加）
- `src/review_predictor/IRL/model/rf_predictor.py` — RF 学習・特徴量抽出（1snap で利用）
- `src/review_predictor/IRL/features/common_features.py` — `FEATURE_NAMES_WITH_PATH` 定義
- `src/review_predictor/IRL/features/path_features.py` — `PathFeatureExtractor`

### 出力

- `outputs/local_rf_w1095/fs_*/` — 1snap 3y 結果（legacy）
- `outputs/local_rf_panel/fs_*/` — 月次パネル 3y 結果（legacy）
- `outputs/local_rf_panel/_cache/` — パネルキャッシュ（legacy）
- `outputs/dynstart_rf_1snap/fs_*/` — **dynstart 1snap 結果（主結果向け）**
- `outputs/dynstart_rf_panel/fs_*/` — **dynstart 月次パネル結果（主結果向け）**
- `outputs/dynstart_rf_panel/_cache/` — dynstart パネルキャッシュ

### 関連ドキュメント

- `writing/HANDOFF_label_definition_audit.md` — A vs B 定義の問題
- `CLAUDE.md` — 全体ルール、特徴量定義、ラベル定義
- `writing/RULES.md` — 論文向けスコア源規約

---

## 8. アーキタイプ別 利得分析（RF 変種 × archetype）

RF の作り方（main 1snap 6m / 長窓 1snap 3y / 月次パネル 3y）を変えても、アーキタイプ別の
IRL/RF 補完がどう変わるかを一括算出するフロー。RF 評価（§2–§4）の下流に位置づく。

### 8.0 流れ（RF 評価のやり直し → アーキタイプ利得まで）

```
(§2/§3) RF 変種を評価
  bash scripts/local/run_rf_only_local.sh   WINDOW_DAYS=1095 OUTBASE=outputs/local_rf_w1095
      → outputs/local_rf_w1095/fs_*/pair_predictions.csv (1snap 3y)
  bash scripts/local/run_rf_panel_local.sh
      → outputs/local_rf_panel/fs_*/rf_panel_predictions.csv (月次パネル 3y)
        │
        ▼
(§8.2) アーキタイプ利得分析  →  writing/tmp/scripts/rf_variant_archetype_gain.py
        ├─ B ラベルを combined_raw から再計算（承諾予測。§1.2 参照）
        ├─ IRL 確率 = main volat の irl_dir_prob を共通使用（RF だけ差し替え）
        ├─ archetype 割当 = dev (k4_dev_alpha dominant A1–A4) / pair (k4_pair_alpha dominant P1–P4)
        │   ※割当は状態特徴量ベースで label 非依存（再生成不要、HANDOFF 監査で確認）
        └─ 出力: 各 archetype × RF変種 の IRL/RF/アンサンブル AUC、利得(=Ens−RF)、IRL−RF
                 → writing/tmp/results/rf_variant_archetype_{pooled,per_horizon}.csv
```

**評価はすべて B 定義（承諾）で統一**し、候補ペアは main の 4,534 ペアを基準に各変種をマージする
（1snap・月次パネルとも main と同じ 4,534 ペアに完全一致、欠損ゼロを確認済み）。

### 8.1 入力

| 役割 | ソース | 列 |
|---|---|---|
| IRL 確率（legacy 比較用） | `outputs/main32_mce_volat_ep150/lstm_baseline/train_<h>/eval_<h>/pair_predictions.csv` | `irl_dir_prob` |
| **IRL 確率（dynstartB、主推奨）** | `outputs/dynstartB_ep150/lstm_baseline/train_<h>/eval_<h>/pair_predictions.csv` | `irl_dir_prob` |
| RF: main 6m 1snap | 同上 | `rf_dir_prob` |
| RF: 3y 1snap (固定窓、legacy) | `outputs/local_rf_w1095/fs_*/pair_predictions.csv` | `rf_dir_prob` |
| RF: 3y 月次パネル (固定 train_start、legacy) | `outputs/local_rf_panel/fs_*/rf_panel_predictions.csv` | `rf_panel_prob`（`email` 列） |
| **RF: dynstart 1snap (主推奨)** | `outputs/dynstart_rf_1snap/fs_*/pair_predictions.csv` | `rf_dir_prob` |
| **RF: dynstart 月次パネル (主推奨)** | `outputs/dynstart_rf_panel/fs_*/rf_panel_predictions.csv` | `rf_panel_prob`（`email` 列） |
| B ラベル | `data/combined_raw_main32.csv`(+ raw_json) から再計算 | 期間内に1回でも承諾→1 |
| dev archetype | `outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_dev_alpha_with_metrics.csv` | `dominant` (A1–A4) |
| pair archetype | `outputs/analysis/archetype_pair_level_volat_ep150_coldonly/k4_pair_alpha_with_metrics.csv` | `dominant` (→P1–P4) |

### 8.2 再現スクリプトとコマンド

```bash
# RF 変種を新しく評価し直したら、SOURCES のパスだけ書き換えて実行
uv run python writing/tmp/scripts/rf_variant_archetype_gain.py \
    --prediction-time 2023-01-01 \
    --output-dir writing/tmp/results
# 出力: rf_variant_archetype_pooled.csv / rf_variant_archetype_per_horizon.csv + Markdown 表
```

スクリプト先頭の `SOURCES`（`(名前, fmt, base, prob列)` のリスト）を編集すれば変種を追加・差し替え可能。
`fmt` は `volat_main`（`lstm_baseline/train_<h>/eval_<h>` 配置）/ `volat_fs`（`fs_<a>_to_<b>m` 配置）/
`panel_fs`（`email/rf_panel_prob` 形式）の3種に対応。

### 8.3 結果（pooled、B 定義 4,534 ペア、利得 = アンサンブル AUC − RF AUC）

| level | archetype | IRL | main RF / 利得 / IRL−RF | 1snap RF / 利得 / IRL−RF | monthly RF / 利得 / IRL−RF |
|---|---|---|---|---|---|
| pair | **P1 未経験** | .728 | .689 / **+.061** / +.039 | .679 / **+.062** / +.049 | .660 / **+.072** / +.068 |
| dev | A2 レビュー専念 | .766 | .728 / +.035 / +.037 | .757 / +.032 / +.009 | .773 / +.035 / −.008 |
| dev | A4 常時活動 | .788 | .803 / +.020 / −.015 | .789 / +.028 / −.001 | .771 / +.040 / +.017 |
| dev | A3 ベテラン | .766 | .806 / +.021 / −.040 | .791 / +.023 / −.025 | .790 / +.023 / −.024 |
| pair | P4 中程度 | .676 | .714 / +.010 / −.038 | .710 / +.016 / −.034 | .739 / +.013 / −.063 |
| pair | P2 経験低承諾 | .733 | .780 / +.001 / −.047 | .771 / +.006 / −.038 | .759 / +.013 / −.026 |
| pair | P3 経験高承諾 | .709 | .785 / −.002 / −.076 | .781 / −.007 / −.072 | .770 / −.005 / −.061 |
| dev | A1 新人 | .583 | .690 / −.005 / −.106 | .667 / +.005 / −.083 | .710 / −.015 / −.126 |

### 8.4 ホライズン別 利得（Ens−RF） [main / 1snap / monthly]

**開発者archetype**

| dev | 0–3m | 3–6m | 6–9m | 9–12m |
|---|---|---|---|---|
| A1 | −.018/+.004/−.024 | −.038/−.038/−.061 | +.041/+.051/+.025 | +.002/−.008/−.000 |
| A2 | +.036/+.056/−.007 | +.053/+.051/+.064 | +.024/+.016/−.006 | +.047/−.004/+.074 |
| A3 | +.025/+.020/+.020 | +.003/−.005/−.007 | +.032/+.060/+.032 | +.018/+.028/+.038 |
| A4 | +.025/+.037/+.048 | +.006/+.022/+.027 | +.030/+.044/+.044 | +.008/+.010/+.030 |

**ペアarchetype**

| pair | 0–3m | 3–6m | 6–9m | 9–12m |
|---|---|---|---|---|
| P1 | +.062/+.078/+.092 | +.047/+.060/+.037 | +.085/+.085/+.070 | +.032/+.023/+.057 |
| P2 | +.005/+.009/+.009 | −.001/+.011/+.012 | +.009/+.022/+.016 | −.016/−.015/+.004 |
| P3 | −.011/−.015/−.005 | −.034/−.036/−.052 | +.029/+.028/+.014 | +.031/+.022/+.030 |
| P4 | +.028/+.043/−.011 | +.017/+.020/+.018 | +.002/+.005/−.008 | +.000/−.008/+.033 |

### 8.5 ホライズン別 IRL−RF（正=IRL優位） [main / 1snap / monthly]

**開発者archetype**

| dev | 0–3m | 3–6m | 6–9m | 9–12m |
|---|---|---|---|---|
| A1 | −.097/−.053/−.117 | −.230/−.165/−.240 | −.089/−.065/−.132 | −.046/−.108/−.068 |
| A2 | +.012/+.047/−.117 | +.067/+.046/+.049 | −.011/−.029/−.067 | +.146/−.046/+.096 |
| A3 | −.016/−.022/−.024 | −.088/−.065/−.077 | −.035/+.003/−.023 | −.025/−.004/+.007 |
| A4 | −.008/+.012/+.025 | −.040/−.009/−.014 | +.010/+.025/+.028 | −.024/−.025/+.007 |

**ペアarchetype**

| pair | 0–3m | 3–6m | 6–9m | 9–12m |
|---|---|---|---|---|
| P1 | +.039/+.072/+.086 | +.015/+.050/+.024 | +.071/+.073/+.056 | −.001/−.009/+.068 |
| P2 | −.037/−.028/−.029 | −.057/−.033/−.042 | −.024/−.008/−.014 | −.063/−.066/−.036 |
| P3 | −.084/−.090/−.071 | −.127/−.098/−.128 | −.021/−.023/−.032 | −.008/−.018/−.012 |
| P4 | +.020/+.043/−.102 | −.043/−.031/−.058 | −.107/−.091/−.119 | +.001/−.066/−.005 |

### 8.6 読み取り（RF 変種に依らない頑健な構造）

- **どの RF 定式化でも「IRL が勝つ／補完価値が出るのは P1（ディレクトリ未経験）と A2（レビュー専念）」、それ以外（A1 新人・A3 ベテラン・P2/P3 経験ペア）は RF 優位**、という住み分けは不変。
- **P1 が全変種で利得最大**。月次パネル RF は P1 でやや弱い（RF .660）ため IRL の補完余地が最大（利得 +.072、IRL−RF +.068）。短・中期（0–3m, 6–9m）で特に顕著。
- A2 は IRL 寄りだが定式化依存。main は長期 9–12m で IRL−RF +.146 と突出、1snap は長期で RF へ反転、月次は 0–3m で RF が強い。
- RF 単体 AUC は3変種で数pt以内 → **「IRL を足す価値は P1/A2 に集中」という結論は RF の作り方に依存しない**。

### 8.7 新しい実行結果での回し方

1. RF 変種を再評価（§2/§3/§3b のコマンド。`OUTBASE` を新しいパスに）。
2. IRL を新規学習・新ラベルで評価し直した場合は `--main-base` を差し替え。dynstartB に切り替える場合は `outputs/dynstartB_ep150/lstm_baseline/` を指す。
3. アーキタイプを再生成した場合（通常は不要、割当は label 非依存）は `--dev-alpha`/`--pair-alpha` を差し替え。
4. `rf_variant_archetype_gain.py` の `SOURCES` を編集して実行 → §8.3–8.5 と同形式の表が再生成される。

**dynstartB IRL × dynstart RF での再生成 SOURCES 例**:

```python
SOURCES = [
    ("dynstart_1snap",  "volat_fs", "outputs/dynstart_rf_1snap",  "rf_dir_prob"),
    ("dynstart_panel",  "panel_fs", "outputs/dynstart_rf_panel",  "rf_panel_prob"),
    # legacy 比較用に残す場合は以下を追加
    # ("main_6m",        "volat_main", "outputs/main32_mce_volat_ep150/lstm_baseline", "rf_dir_prob"),
    # ("legacy_1snap_3y","volat_fs",   "outputs/local_rf_w1095",   "rf_dir_prob"),
    # ("legacy_panel_3y","panel_fs",   "outputs/local_rf_panel",   "rf_panel_prob"),
]
# IRL 確率の参照先 (--main-base 相当) を outputs/dynstartB_ep150/lstm_baseline に差し替え
```

### 8.8 関連ファイル（§8 追加分）

- `writing/tmp/scripts/rf_variant_archetype_gain.py` — 本節の再現スクリプト
- `writing/tmp/results/rf_variant_archetype_pooled.csv` — pooled 結果
- `writing/tmp/results/rf_variant_archetype_per_horizon.csv` — ホライズン別（利得 + IRL−RF）
- `writing/tmp/data/pair_predictions_b/` — B ラベル付き per-cell（参考）
- `writing/tmp/AB_ANALYSIS_SUMMARY.md` — main RF での A vs B アーキタイプ補完性の総括

# 引き継ぎ: ラベル定義の非対称と論文分析全体への影響監査

**作成日**: 2026-06-02
**コンテキスト**: 月次パネル RF の検証中に、`pair_predictions.csv` の `label` 列と RF/IRL 訓練時のラベル定義が**非対称**であることが判明。論文の `writing/analysis/` 配下の **40+ 分析スクリプト** が下流で影響を受ける可能性がある。

---

## TL;DR

- **volat 訓練** (`rf_predictor.py:154-157`, IRL 同様): 期間内に **1回でも承諾 → 1**、全拒否 or 依頼なし → 0（**B 定義**）
- **volat 評価** (`eval_mce_irl_path_prediction.py:1229` + `get_directory_developers`): 期間内に **dir に行が存在すれば → 1**（label の値無視。承諾も拒否も同じ扱い、**A 定義**）
- IRL/RF 間の比較は volat 内で公正（同じ A 定義で評価）
- ただし「論文中の AUC は『dev が dir に**アサインされるか**』を測ってる」ことになり、**ユーザの直感（『承諾するか』を予測したい）と乖離**
- 同じ volat モデルを B 定義で再評価すると **AUC_ROC はむしろ上がる**（0.81→0.84 など）。lift で見るともっと顕著
- 下流分析 (`label_<h>`、`pair_long_with_archetype.csv` 等) は **全て A 定義の label を継承**

---

## 1. 問題の正確な定義

### 1.1 訓練ラベル（B 定義）

#### RF_Dir 訓練 (`src/review_predictor/IRL/model/rf_predictor.py:147-157`)

```python
for directory in dev_dirs:
    dev_eval = label_df[
        (label_df["email"] == email)
        & (label_df[dirs_column].map(
            lambda ds: directory in ds if ds else False
        ))
    ]
    if len(dev_eval) == 0:
        label = 0                                       # 依頼なし → 0
    else:
        label = 1 if (dev_eval["label"] == 1).any() else 0  # 1回でも承諾 → 1、全拒否 → 0
```

#### IRL 訓練 (`scripts/train/train_model.py:266-313` 系、`CLAUDE.md` にも明記)

- 正例（label=1, weight=1.0）: 期間内にレビュー依頼を受けて1つでも承諾
- 負例（label=0, weight=1.0）: 期間内に依頼を受けたが全拒否
- 弱負例（label=0, weight=0.1）: 期間内に依頼なし、拡張期間に依頼あり
- 除外: 拡張期間にも依頼なし

→ いずれも「1回でも承諾」が正例条件（B 定義）

### 1.2 評価ラベル（A 定義）

#### pair_predictions.csv の `label` 列 (`scripts/analyze/eval/eval_mce_irl_path_prediction.py:1218-1238`)

```python
pair_rows: List[Dict] = []
for d, devs in dir_developers.items():
    actual_devs = actual_dir_devs.get(d, set())        # ← :752
    for dev in devs:
        row = {
            "developer": dev,
            "directory": d,
            "label": 1 if dev in actual_devs else 0,   # ← ココ
            "irl_dir_prob": ...,
            "rf_dir_prob": ...,
            ...
        }
        pair_rows.append(row)
```

`actual_devs` の生成 (`:752`):

```python
actual_dir_devs = get_directory_developers(df, future_start, future_end)
```

`get_directory_developers` の実装 (`src/review_predictor/IRL/features/directory_contributors.py:37-49`):

```python
mask = (df["timestamp"] >= start_time) & (df["timestamp"] < end_time)
sub = df.loc[mask]
result: Dict[str, Set[str]] = {}
for _, row in sub.iterrows():
    email = row.get("email")
    dirs = row.get(dirs_column)
    if not email or not dirs:
        continue
    for d in dirs:
        if d == ".":
            continue
        result.setdefault(d, set()).add(email)    # ← label 値を一切見ない
return result
```

→ **承諾も拒否も同じく positive 扱い**

### 1.3 実証検証

`outputs/main32_mce_volat_ep150/lstm_baseline/train_0-3m/eval_0-3m/pair_predictions.csv` で:

- volat の `label=1` 数 = 1841
- any関与 (A) で説明 = **1841/1841 = 100%**
- 承諾あり (B) で説明 = 1225/1841 = **67%**

→ 確定: volat eval は **A 定義**

---

## 2. 同一モデルを B 定義で再評価した結果

`/tmp/recompute_volat_b.py` で既存 `pair_predictions.csv` の prob 列 + B-relabel した actual で AUC 再計算。

| h | def | base_rate | IRL_Dir ROC/PR | RF_Dir ROC/PR | Ensemble ROC/PR |
|---|---|---|---|---|---|
| 0-3 | A (現状) | 0.406 | 0.779/0.745 | 0.792/0.773 | 0.813/0.793 |
| 0-3 | **B** | 0.270 | **0.809**/0.654 | **0.821**/0.705 | **0.842**/0.718 |
| 3-6 | A | 0.353 | 0.770/0.697 | 0.804/0.740 | 0.819/0.758 |
| 3-6 | **B** | 0.225 | **0.801**/0.597 | **0.857**/0.701 | **0.864**/0.695 |
| 6-9 | A | 0.326 | 0.774/0.670 | 0.789/0.686 | 0.816/0.719 |
| 6-9 | **B** | 0.217 | **0.787**/0.560 | **0.818**/0.587 | **0.840**/0.615 |
| 9-12 | A | 0.274 | 0.779/0.622 | 0.793/0.650 | 0.815/0.680 |
| 9-12 | **B** | 0.179 | 0.780/0.484 | 0.803/0.532 | 0.815/0.551 |

**観察**:

1. **B 定義の方が AUC_ROC は高い**（IRL/RF/Ensemble 全 horizon で）
   - A 定義は「全拒否のみ」も正例に入れる→borderline 例多く識別困難
   - B 定義は「承諾あり」のみ正例→分離クリア
2. AUC_PR は base_rate が下がる分絶対値も下がる
3. **lift (PR / base_rate)** で比較すると B の方が顕著（例: 0-3m Ensemble、A 1.94× → B 2.66×）
4. IRL/RF/Ensemble の **相対順位は維持**（Ensemble > RF ≥ IRL）

→ **論文の結論（IRL+RF Ensemble が最強）は B 定義に変えても崩れない**

---

## 3. 影響を受ける下流分析（全て A 定義の label を使用）

### 3.1 pair_predictions.csv → pair_trajectories.csv の伝搬

`writing/analysis/scripts/00_regenerate_pair_trajectories.py:31-32`:

```python
d = d.rename(columns={
    "label": f"label_{h}",        # ← A 定義の label をそのまま label_{h} に
    ...
})
```

出力: `writing/analysis/data/pair_trajectories.csv` (および派生の `pair_long_with_archetype.csv` 等)

### 3.2 `label_<h>` を使う論文分析 (~40+ ファイル)

`grep -l 'label_0-3m\|label_3-6m\|label_6-9m\|label_9-12m' writing/analysis/scripts/*.py` の結果（一部）:

- **パターンクラス分析**: `14_pattern_class_auc.py`, `13_pattern_class_rank_winrate.py`
  - 4 horizon のラベルから 16 パターン (0000, 0001, ..., 1111) を構築
  - **パターン定義自体が A 定義依存** → B にすると 1111 (常に positive) の dev が減る
- **アーキタイプ分析**: `22_archetype_x_pattern.py`, `05_dir_archetype_distribution.py`, `05b/c/d/e/f`, `archetype_*` 系
- **ペア vs ベースライン**: `16_pair_vs_baseline.py`
- **連続値層別 AUC**: `15_activity_stratified_auc.py`, `17_continuous_stratified_auc.py`
- **メカニズム分析**: `09_irl_residual_explain.py`, `10_fluctuating_dev_decomp.py`, `30b_a4_mechanism.py`, `30c_p1_mechanism.py`
- **デ ィレクトリ予測可能性**: `03_directory_predictability.py`, `07_all_dirs_predictability.py`, `12_horizon_dir_attrition.py`
- **多変量・回帰**: `18_irl_gain_multivar.py`, `21_wilcoxon_tests.py`, `26_interval_rigor.py`
- **その他**: `01_disagreement_analysis.py`, `02_horizon_feature_importance.py`, `04/04b_biorhythm_*`, `06_switch_rate_vs_accuracy.py`, `11_horizon_archetype_irl_rescue.py`, `19_case_study_irl_rf.py`, `20_topk_recommendation.py`, `24_a2_inactive_probe.py`, `25_archetype_delta_significance.py`, `27_directory_master_table.py`, `29_project_master_table.py`, `31a/32a/32c_p1_*`, `33a/b/c/d_*`, `34_irl_alone_fluctuation.py`, `36_horizon_3_6m_probe.py`, `37_feature_reliance_irl_rf.py`, `verify_ens_decomp*`

完全リスト取得コマンド:
```bash
grep -lE 'label_0-3m|label_3-6m|label_6-9m|label_9-12m' writing/analysis/scripts/*.py
```

### 3.3 archetype 生成元 (`outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/`)

- `k4_dev_alpha_with_metrics.csv` などのアーキタイプ α は `archetype_irl_rf.py` (in `scripts/analyze/study/`) で生成
- このスクリプトも pair_predictions.csv の label を使用 → **アーキタイプ割り当て自体が A 定義依存の可能性大**
- 要確認: B 定義でアーキタイプ生成し直すと dev のクラスタリングが変わるか

---

## 4. 検証方針（新 Claude タスク）

### 4.1 まず決めること（人間の判断）

- [ ] **論文で採用するラベル定義は A か B か**
  - A: 現状維持（「アサイン予測」task）→ 再 run 不要、ただし論文内で task 定義を明示
  - B: 「承諾予測」task に変更 → 全下流分析を再 run
  - 推奨は **B**（ユーザの直感に合う + lift で見るとモデル性能の方が顕著に出る + AUC_ROC も上がる）

### 4.2 B 採用の場合の手順

#### Step 1: pair_predictions.csv に label_B を追加

```python
# scripts/analyze/study/patch_pair_predictions_with_label_b.py (新規作成)
# 既存 pair_predictions.csv に label_B 列を追加保存
# あるいは label を上書きするバージョン
```

入力: `outputs/main32_mce_volat_ep150/lstm_baseline/train_<h>/eval_<h>/pair_predictions.csv`
出力: 同パスに上書き、または `pair_predictions_b.csv` として並列保存

実装の参考: `/tmp/recompute_volat_b.py` の `accept_pairs()` 関数

#### Step 2: pair_trajectories.csv を B 定義で再生成

```python
# writing/analysis/scripts/00_regenerate_pair_trajectories.py を改修
# d.rename({"label": f"label_{h}"}) を label_B ベースに
```

#### Step 3: archetype 生成をやり直す

```bash
# 既存の archetype 生成スクリプト (scripts/analyze/study/archetype_irl_rf.py 等) を
# B-relabeled pair_predictions で実行
# 出力先: outputs/analysis/archetype_irl_rf_volat_ep150_coldonly_B/
```

→ k4 dev alpha が変わる可能性大 → アーキタイプの解釈も変わる

#### Step 4: 論文分析スクリプトの再実行

優先度高:
1. `14_pattern_class_auc.py` — パターンクラス AUC
2. `16_pair_vs_baseline.py` — pair AUC 主表
3. `15_activity_stratified_auc.py`, `17_continuous_stratified_auc.py` — 層別 AUC
4. `13_pattern_class_rank_winrate.py` — IRL vs RF rank 勝率
5. `22_archetype_x_pattern.py` — archetype × pattern クロス分析
6. `21_wilcoxon_tests.py` — 統計検定

優先度中:
7. `03_directory_predictability.py`, `07_all_dirs_predictability.py`
8. `09_irl_residual_explain.py`, `10_fluctuating_dev_decomp.py`
9. `05_dir_archetype_distribution.py` + 派生 (05b/c/d/e/f)

優先度低（影響軽微の可能性）:
10. それ以外

#### Step 5: 図と数値の差分確認

各分析の出力 (`writing/analysis/results/*.{csv,json,pdf}`) を A 版とバージョン比較し、論文文中の数値も更新。

### 4.3 B 採用でも A 結果は保存しておく

論文の説明として「我々は『承諾予測』を主タスクとするが、参考までに『アサイン予測』（既存研究の標準的設定）でも比較する」と書くなら、A 定義の結果は付録として保持。

---

## 5. 現状サマリ

### 5.1 完了済み

- ✅ ラベル定義の非対称を発見・確定
- ✅ 影響範囲（40+ analyse スクリプト）を洗い出し
- ✅ 同じ volat モデルで B 定義 AUC 再計算（全 horizon、IRL/RF/Ensemble）
- ✅ 月次パネル RF (`scripts/analyze/eval/eval_rf_panel.py`, `scripts/local/run_rf_panel_local.sh`) は B 定義で実装 → B 版 RF のリファレンス値あり
  - 出力: `outputs/local_rf_panel/fs_{0_3,3_6,6_9,9_12}m/summary_metrics.json` + `rf_panel_predictions.csv`

### 5.2 未着手

- ⬜ 論文採用ラベル定義の最終決定（人間判断）
- ⬜ `pair_predictions.csv` の B-label 追加（Step 1）
- ⬜ `pair_trajectories.csv` の B 版生成（Step 2）
- ⬜ archetype 再生成（Step 3）
- ⬜ 下流 40+ 分析の再実行（Step 4）
- ⬜ 論文中の数値・図の更新

### 5.3 関連ファイル

- `CLAUDE.md` — 訓練時のラベル定義（B）が記載されてる
- `writing/RULES.md` — 結果セットとスコア源の規約（要確認）
- `/tmp/recompute_volat_b.py` — クイック検証スクリプト（参考実装）
- `outputs/main32_mce_volat_ep150/lstm_baseline/train_*/eval_*/pair_predictions.csv` — 既存 (A 定義) のペア予測
- `outputs/local_rf_panel/fs_*/rf_panel_predictions.csv` — B 定義の月次パネル RF 予測

---

## 6. 次の Claude セッションへの依頼テンプレ

```
このリポジトリ /Users/kazuki-h/Master_research で論文 (writing/) の
ラベル定義監査を引き継いでほしい。

writing/HANDOFF_label_definition_audit.md を読んで、
4.2 Step 1-2 (pair_predictions.csv に label_B 追加 → pair_trajectories.csv
B 版生成) をまず実装してくれ。

判断保留点 (4.1) はユーザに確認すること。
```

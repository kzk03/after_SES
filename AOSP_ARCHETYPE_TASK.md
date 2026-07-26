# AOSP: RF/GBDT完了 → 原型分析 実行手順

## 背景・ゴール

SES2026カメラレディ論文(`SES/ses_main.tex`)に、外的妥当性検証としてAOSPのケーススタディを追加中。
OpenStack(32リポジトリ)と同一のパイプライン・評価設定でAOSPを分析し、以下をTeX中の`\todo{}`プレースホルダに埋める:

- 対象リポジトリ数・データ規模(change数・依頼行数・レビュアー数): `ses_main.tex` 452行目
- AOSP版 IRL/RF AUC-ROC(4評価期間): 464行目
- 原型分析: 原型数Kの選択根拠(957行目)、ペア単位・開発者単位それぞれ4原型の解釈名・特徴・評価期間別AUC(964〜1092行目のプレースホルダ表・図)、OpenStackとの比較考察(1095行目)

`ses_main.tex`の577〜929行目に、OpenStackで実施済みの原型分析の実例(表・図・考察文)がある。AOSPでも同じ深さ・同じ形式の結果を作るのが目標。

## 現在の状態(2026-07-26時点)

- **IRL (`outputs/aosp32/dynstartB_ep150/lstm_baseline/`)**: 4評価期間(train_0-3m〜train_9-12m)すべて完了。新定義対応済み。
- **RF/GBDT v1 (`outputs/aosp32/rf_gbdt_1snap/`)**: 旧定義。**archive扱い、使わない。**
- **RF/GBDT v2 (`outputs/aosp32/rf_gbdt_1snap_v2/`)**: 新定義。`fs_0_to_3m`・`fs_3_to_6m`・`fs_9_to_12m`は完了。**`fs_6_to_9m`が未完了**(実行中断あり、要再実行)。

実行前に必ず `find outputs/aosp32/rf_gbdt_1snap_v2 -name summary_metrics.json` で最新状態を確認すること(このドキュメント作成後に進んでいる可能性あり)。

## 重要な注意点(ハマりどころ)

1. **`rf_gbdt_1snap`(v1)は使わない**。必ず`rf_gbdt_1snap_v2`を参照する。
2. **HUB_SCORES_CSV/COCHANGE_NEIGHBORS_CSVはファイル名に`main32`が残っているが中身はAOSP用**。実体は `experiments/dependency_analysis/results_aosp/hub_scores_main32.csv` と `experiments/dependency_analysis/results_aosp/cochange_neighbors_main32.csv`(`results/`ではなく`results_aosp/`ディレクトリ)。OpenStack用の`experiments/dependency_analysis/results/hub_scores_main32.csv`と混同しないこと。
3. **`DIR_PREFIX_PROJECT=1` は必須**。AOSPは32リポジトリ間でディレクトリ名が衝突するため、これを付けないとpath特徴量(ディレクトリ親和度)が壊れる。
4. **`COCHANGE_MIN_WEIGHT=2 COCHANGE_TOPK=100`も必ず揃える**。特徴量の定義に関わるパラメータなので、他の完了済み期間と条件を一致させる。
5. `--prediction-time 2023-01-01 --train-end 2022-01-01` はOpenStackと同一値(ユーザー確認済み)。

## Step 1: RF/GBDT v2の残り期間(fs_6_to_9m)を完了する

```
REVIEWS=data/combined_raw_aosp32.csv RAW_JSON_LIST_FILE=data/combined_raw_aosp32.raw_json_list.txt HUB_SCORES_CSV=experiments/dependency_analysis/results_aosp/hub_scores_main32.csv COCHANGE_NEIGHBORS_CSV=experiments/dependency_analysis/results_aosp/cochange_neighbors_main32.csv DIR_PREFIX_PROJECT=1 COCHANGE_MIN_WEIGHT=2 COCHANGE_TOPK=100 GBDT_DIR=hist,lgbm,xgb OUTBASE=outputs/aosp32/rf_gbdt_1snap_v2 N_JOBS=12 HORIZONS="6" nohup bash scripts/local/run_rf_dynstart_1snap_local.sh > logs/aosp32_rf_v2_fs6.log 2>&1 &
```

`tail -f logs/aosp32_rf_v2_fs6.log` で進行確認。`outputs/aosp32/rf_gbdt_1snap_v2/fs_6_to_9m/summary_metrics.json` が生成されたら完了。

完了確認:
```
find outputs/aosp32/rf_gbdt_1snap_v2 -name summary_metrics.json | sort
```
4つ(fs_0_to_3m, fs_3_to_6m, fs_6_to_9m, fs_9_to_12m)揃っていることを確認する。

## Step 2: データセット統計を取得(論文452行目用)

```
uv run python scripts/pipeline/verify_aosp_collection.py
```
末尾の `合計: X changes / Y 依頼行` を控える。

レビュアー数(ユニーク開発者数)は `data/combined_raw_aosp32.csv` の該当列(まず `head -1 data/combined_raw_aosp32.csv` で列名を確認。`data/combined_raw_main32.csv` と同一スキーマのはずなので、OpenStack側でreviewer/developerに相当する列名を確認してから同じ列でnuniqueを取る)。

## Step 3: IRL/RF AUC比較表(論文464行目用)

Step1完了後、各期間の `summary_metrics.json` から `IRL_Dir.clf_auc_roc`(IRL側は `dynstartB_ep150/lstm_baseline/train_*/eval_*/summary_metrics.json` の `IRL_Dir`)と `RF_Dir.clf_auc_roc` / `GBDT_Dir.clf_auc_roc`(RF側は `rf_gbdt_1snap_v2/fs_*/summary_metrics.json`)を4期間分抽出し、`ses_main.tex`の表`tab:rq1`と同じ形式(1-3m, 4-6m, 7-9m, 10-12m + 4期間平均)でまとめる。

## Step 4: 原型分析を実行

Step1完了後に実行(内部で4期間すべての`rf_gbdt_1snap_v2/fs_*/pair_predictions.csv`を読むため、揃っていないとエラーになる):

```
DIR_PREFIX_PROJECT=1 COCHANGE_MIN_WEIGHT=2 COCHANGE_TOPK=100 uv run python scripts/analyze/study/t0_archetype.py --irl-base outputs/aosp32/dynstartB_ep150/lstm_baseline --rf-base outputs/aosp32/rf_gbdt_1snap_v2 --prediction-time 2023-01-01 --train-end 2022-01-01 --output-dir outputs/aosp32/archetype --level both --k-grid 2 3 4 5 6 --data data/combined_raw_aosp32.csv --raw-json-list data/combined_raw_aosp32.raw_json_list.txt --raw-json-glob "data/aosp/raw_json/aosp__*.json" --hub-scores experiments/dependency_analysis/results_aosp/hub_scores_main32.csv --cochange-neighbors experiments/dependency_analysis/results_aosp/cochange_neighbors_main32.csv
```

出力先: `outputs/aosp32/archetype/`(elbow用RSS推移、K=4選択時の原型別特徴量プロファイル、原型×評価期間のAUC表 相当のファイルが出るはず。具体的なファイル名・形式は既存のOpenStack向け出力 `outputs/t0_sweep/*/archetype/` を参照して揃える)。

## Step 5: 論文用にまとめる

以下を成果物としてまとめてほしい(`ses_main.tex`の対応箇所を埋められる形式):

1. Kの選択(elbowのRSS推移とその根拠、1〜2文)
2. ペア単位4原型: 各原型の解釈名・特徴量に基づく説明・n数・評価期間別AUC(IRL/RF/Δ)
3. 開発者単位4原型: 同上
4. OpenStackの原型構成(P1〜P4: 他プロジェクト経験者/関与低下型/高承諾型/低活動・低承諾型、A1〜A4: 低活動型/短期・レビュー中心型/高承諾・高活動型/常時活動・低承諾型 ※`ses_main.tex`772-786, 835-844行目参照)と比べて、同様の傾向か異なる原型構成かを1〜2段落で考察

原型の解釈名は原型分析の出力(特徴量プロファイル)を見て提案してよいが、最終的な採否はユーザー判断。数値は必ず実際の出力ファイルから転記し、推測で書かない。

# SES — 論文モデル再現 + horizon別「動的学習開始日」

論文モデル `main32_mce_volat_ep150`（MCE-IRL, 4 horizon）を**再現**し、その上で
**horizon ごとに学習開始日を早めて特徴量計測期間を揃える（②）**実験を行う自己完結ディレクトリ。

- コードは `tmp_thesis/`（凍結スナップショット）から複製：`src/`, `scripts/`。
- 大容量入力は symlink：`data/ → ../data`, `experiments/ → ../experiments`。
- 出力は `SES/outputs/` 配下に生成（凍結物は上書きしない）。
- **実行は必ず CWD=SES root**（raw_json リストが相対パスのため）。

## 背景（なぜ②か）
学習はリーク防止で各月次ステップを `month_end + FE ≤ train_end(2022-01)` にクリップするため、
将来窓 FE が長い horizon ほど最新 month_end が早まり、**特徴量を計測する月次ステップ数が horizon 間でズレる**：

| horizon | FE | 最後の month_end | ステップ数 |
|---|---|---|---|
| 0-3m | 3 | 2021-10 | **33** |
| 3-6m | 6 | 2021-07 | 30 |
| 6-9m | 9 | 2021-04 | 27 |
| 9-12m | 12 | 2021-01 | **24** |

②＝`train_start(h) = 2019-01-01 − FS(h)ヶ月` と horizon 別に早め、後方で失う分を前方で補い**全 horizon 33 ステップ**に揃える
（0-3m:2019-01 / 3-6m:2018-10 / 6-9m:2018-07 / 9-12m:2018-04。データは 2012-06 まであり実行可能）。

## A/B の違い（学習側は共通、eval 窓だけ違う）
- **B案（既定）**: 学習を 33 に揃える。**eval は 24ヶ月のまま**（論文準拠・乖離小）。`EVAL_DIR_HISTORY_MONTHS=24`
- **A案（代替）**: 学習を 33 に揃え、**eval も 33ヶ月**にして窓長を整合。`EVAL_DIR_HISTORY_MONTHS=33`

実装は driver の env フラグで切替（コードは共通）：
- `DYNAMIC_START` 0/1（学習側 horizon 別開始）
- `EVAL_DIR_HISTORY_MONTHS` 24/33（eval の IRL 月次履歴窓 = `prediction_time − Nヶ月`）

## 実行コマンド（CWD=SES）
共通 env（path 特徴の hub/cochange CSV は再現に必須）：
```bash
export HUB_SCORES_CSV=experiments/dependency_analysis/results/hub_scores_main32.csv
export COCHANGE_NEIGHBORS_CSV=experiments/dependency_analysis/results/cochange_neighbors_main32.csv
export EPOCHS=150 PATIENCE=15
export DEVICE=cuda   # local は mps か cpu
```

**Phase A（再現・固定2019開始・eval24）**
```bash
DYNAMIC_START=0 EVAL_DIR_HISTORY_MONTHS=24 CACHE_TAG=main32_volat \
bash scripts/variant/run_mce_irl_variant_single.sh 0 lstm_baseline outputs/reproA_ep150 0
```

**Phase B（②・B案＝既定）**
```bash
DYNAMIC_START=1 EVAL_DIR_HISTORY_MONTHS=24 CACHE_TAG=main32_volat_dynstartB \
bash scripts/variant/run_mce_irl_variant_single.sh 0 lstm_baseline outputs/dynstartB_ep150 0
```

**A案（代替・検証用）**
```bash
DYNAMIC_START=1 EVAL_DIR_HISTORY_MONTHS=33 CACHE_TAG=main32_volat_dynstartA \
bash scripts/variant/run_mce_irl_variant_single.sh 0 lstm_baseline outputs/dynstartA_ep150 0
```

> CACHE_TAG をフェーズで分けること（同名キャッシュ再利用＝動的開始が無効化する silent bug）。
> 本実装ではキャッシュ名にも train_start を含める（`mce_traj_<win>_ts<start>.pkl`）ので二重に安全。

## 検証（特徴量が論文と一致するか）
**参照軌跡キャッシュは存在しない**ため、凍結モデル出力 `tmp_thesis/outputs/main32_mce_volat_ep150/lstm_baseline/` を anchor に：
- **gate(b) 安価・再学習不要**: `--irl-dir-model` に凍結 `.pt` を指定して eval 再実行 → `pair_predictions.csv` の `irl_dir_prob` を `(developer,directory)` 結合照合。SES の特徴量コードが eval 時に一致するか確認（同一HW |Δ|<1e-5 / HW跨ぎ <1e-3）。
- **gate(c) 秒**: 生成キャッシュ `mce_traj_*.pkl` の `step_actions∈{0,1}`・state_dim=32・各 window の最終 month_end を assert。
- **(a) 権威・要再学習**: Phase A の `summary_metrics.json`/`pair_predictions` を凍結と照合（まず 0-3m）。

## ②の等化チェック（成果物）
Phase A/B のキャッシュで window 別の中央ステップ数を表化：Phase A は 9-12m が 0-3m より ~9 少 → Phase B で 4 window 差 ≤1 を確認。

## 既知の食い違い（要認識）
- eval のコメントは「訓練 2021-01~2023-01(24ヶ月)」と書くが、実 driver は `train_start=2019`（学習は最大33ヶ月）。
- 論文本文は「評価 2022-2023・t0=2022-01-01」だが driver は `EVAL_CUTOFF=2023-01-01`（評価は 2023-）。
- → 再現は **driver 設定（学習2019-2022 / 評価2023-）が正**。
- 学習(累積、開始2019 or 動的) と eval(既定24ヶ月=開始2021) は累積起点が異なる。A案はこれを 33 で揃える方向。

## デバイス
- 重い学習（4×150 epoch）は **CUDA サーバ推奨**。本 Mac は cuda 無し / mps あり → `DEVICE=mps`（or cpu）で動くが低速、かつ凍結（CUDA学習）とは byte 不一致（HW跨ぎ tol で照合）。
- gate(b)（凍結重みでの eval/推論）は local でも可。

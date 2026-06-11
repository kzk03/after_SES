# Directory-level Prediction Trend Analysis

## 実行

```bash
uv run python scripts/analyze/study/per_dir_analysis.py
```

## データ概要
- 入力: `outputs/main32_mce_volat_ep150_raw/lstm_baseline/train_W/eval_W/pair_predictions.csv` (W in ['0-3m', '3-6m', '6-9m', '9-12m'])
- ディレクトリ (モジュール)数: 390 件
- 集計対象 (auc_reliable, n_pos >= 30 & n_neg >= 30): 46 件
- 内訳: project-specific 36 / cross-cutting 10

## 主要発見

### H1: サンプル数と AUC の相関
- Spearman ρ(log n_pair, Ens AUC) = **-0.022** (p = 0.884) → ほぼ相関なし

### H2: project-specific vs cross-cutting の Ens AUC 差
- Mann-Whitney U = 234, p = 0.154
- 中央値: project-specific 0.842 vs cross-cutting 0.778
- 判定: 有意差なし

### H3: Ensemble の単独優位性 (directory 横断)
- Ens vs IRL: W = 96, p_bonf = 1.97e-07, r = 0.696
- Ens vs RF:  W = 307, p_bonf = 0.0199, r = 0.261

## Top 5 / Bottom 5 directories

### Top 5 by Ens AUC

| directory | category | n_pair | IRL | RF | Ens | gain |
|---|---|---:|---:|---:|---:|---:|
| neutron/objects | project-specific | 76 | 0.787 | 0.922 | **0.919** | -0.003 |
| octavia/tests | project-specific | 156 | 0.902 | 0.903 | **0.918** | +0.015 |
| neutron/conf | project-specific | 88 | 0.837 | 0.868 | **0.914** | +0.047 |
| neutron/api | project-specific | 64 | 0.848 | 0.827 | **0.912** | +0.064 |
| neutron/plugins | project-specific | 232 | 0.833 | 0.904 | **0.901** | -0.003 |

### Bottom 5 by Ens AUC

| directory | category | n_pair | IRL | RF | Ens | gain |
|---|---|---:|---:|---:|---:|---:|
| cinder/image | project-specific | 112 | 0.556 | 0.661 | **0.615** | -0.046 |
| tacker/sol_refactored | project-specific | 84 | 0.561 | 0.738 | **0.686** | -0.051 |
| tacker/tests | project-specific | 116 | 0.613 | 0.764 | **0.703** | -0.061 |
| devstack | cross-cutting | 292 | 0.727 | 0.678 | **0.721** | -0.007 |
| nova/scheduler | project-specific | 64 | 0.700 | 0.653 | **0.731** | +0.031 |

## 出力ファイル
- `directory_window_metrics.csv` : directory × window の生メトリクス
- `directory_pooled_metrics.csv` : directory 単位 4 window pool + bootstrap CI
- `stats_test.json` : 検定結果 (Spearman, Mann-Whitney, Wilcoxon)
- `fig_directory_auc_ranked.pdf` : Top/Bottom の AUC ranked bar
- `fig_category_violin.pdf` : project-specific vs cross-cutting の violin
- `fig_n_vs_auc_scatter.pdf` : log(n_pair) vs Ens AUC の scatter

## 解釈方針
- **auc_reliable=False** の単位 (n_pos<30 or n_neg<30) は信頼区間が広く
  個別解釈は避ける。Top/Bottom 集計でも除外している。
- **project-specific category** は main_repos 32 と一致する top_dir 名
  (例: nova, neutron, cinder)。**cross-cutting** は repo を跨いで共有される
  ディレクトリ (例: releasenotes, doc, etc, tools)。
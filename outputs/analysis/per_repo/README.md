# Directory-level Prediction Trend Analysis

## 実行

```bash
uv run python scripts/analyze/study/per_dir_analysis.py
```

## データ概要
- 入力: `outputs/main32_mce_volat_ep150_raw/lstm_baseline/train_W/eval_W/pair_predictions.csv` (W in ['0-3m', '3-6m', '6-9m', '9-12m'])
- リポジトリ (top_dir)数: 69 件
- 集計対象 (auc_reliable, n_pos >= 30 & n_neg >= 30): 27 件
- 内訳: project-specific 17 / cross-cutting 10

## 主要発見

### H1: サンプル数と AUC の相関
- Spearman ρ(log n_pair, Ens AUC) = **0.249** (p = 0.211) → 弱い相関のみ

### H2: project-specific vs cross-cutting の Ens AUC 差
- Mann-Whitney U = 83, p = 0.94
- 中央値: project-specific 0.810 vs cross-cutting 0.766
- 判定: 有意差なし

### H3: Ensemble の単独優位性 (directory 横断)
- Ens vs IRL: W = 36, p_bonf = 0.000146, r = 0.630
- Ens vs RF:  W = 47, p_bonf = 0.000602, r = 0.704

## Top 5 / Bottom 5 repositories

### Top 5 by Ens AUC

| top_dir | category | n_pair | IRL | RF | Ens | gain |
|---|---|---:|---:|---:|---:|---:|
| octavia | project-specific | 696 | 0.874 | 0.871 | **0.892** | +0.018 |
| zuul.d | cross-cutting | 180 | 0.830 | 0.853 | **0.873** | +0.020 |
| designate | project-specific | 292 | 0.842 | 0.801 | **0.862** | +0.020 |
| doc | cross-cutting | 1196 | 0.800 | 0.827 | **0.849** | +0.022 |
| swift | project-specific | 224 | 0.828 | 0.831 | **0.845** | +0.014 |

### Bottom 5 by Ens AUC

| top_dir | category | n_pair | IRL | RF | Ens | gain |
|---|---|---:|---:|---:|---:|---:|
| magnum | project-specific | 244 | 0.618 | 0.570 | **0.607** | -0.011 |
| mistral | project-specific | 300 | 0.621 | 0.664 | **0.670** | +0.006 |
| playbooks | cross-cutting | 216 | 0.596 | 0.658 | **0.672** | +0.015 |
| trove | project-specific | 108 | 0.785 | 0.642 | **0.704** | -0.080 |
| glance | project-specific | 216 | 0.675 | 0.711 | **0.708** | -0.004 |

## 出力ファイル
- `top_dir_window_metrics.csv` : top_dir × window の生メトリクス
- `top_dir_pooled_metrics.csv` : top_dir 単位 4 window pool + bootstrap CI
- `stats_test.json` : 検定結果 (Spearman, Mann-Whitney, Wilcoxon)
- `fig_top_dir_auc_ranked.pdf` : Top/Bottom の AUC ranked bar
- `fig_category_violin.pdf` : project-specific vs cross-cutting の violin
- `fig_n_vs_auc_scatter.pdf` : log(n_pair) vs Ens AUC の scatter

## 解釈方針
- **auc_reliable=False** の単位 (n_pos<30 or n_neg<30) は信頼区間が広く
  個別解釈は避ける。Top/Bottom 集計でも除外している。
- **project-specific category** は main_repos 32 と一致する top_dir 名
  (例: nova, neutron, cinder)。**cross-cutting** は repo を跨いで共有される
  ディレクトリ (例: releasenotes, doc, etc, tools)。
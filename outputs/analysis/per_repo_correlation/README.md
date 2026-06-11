# Repo 特性 × AUC の相関・回帰分析

`per_repo_basic_stats/project_stats.csv` の repo 特性と
`per_repo/top_dir_pooled_metrics.csv` の AUC を merge し、
どの特性が予測精度を説明するかを定量化する。

## 対象: 26 repo (基本統計と AUC の両方が揃うもの)

## 主要相関 (Spearman)

AUC を予測する特徴量との相関 (* = p<0.05):

| 特徴量 | vs IRL AUC | vs RF AUC | vs Ens AUC | vs gain |
|---|---:|---:|---:|---:|
| n_review | **+0.578*** | **+0.606*** | **+0.770*** | **+0.579*** |
| accept_rate | **+0.458*** | +0.370 | **+0.416*** | +0.122 |
| n_unique_dev | **+0.557*** | **+0.508*** | **+0.683*** | **+0.535*** |
| reviews_per_dev | **+0.455*** | **+0.579*** | **+0.699*** | **+0.521*** |
| median_monthly_unique_acc_dev | **+0.605*** | **+0.630*** | **+0.755*** | **+0.527*** |

### 解釈
- **承諾率 → Ens AUC**: ρ=+0.416 (p=0.0345) → 承諾率が高い repo ほど予測しやすい
- **n_review → Ens AUC**: ρ=+0.770 (p=4.32e-06) → 規模 (レビュー件数) が大きいほど予測しやすい

## 線形回帰: 3 特徴量で AUC を予測

特徴量: log10(n_review+1), accept_rate, log10(n_unique_dev+1)

| target | R² | intercept | β log_n_review | β accept_rate | β log_n_dev |
|---|---:|---:|---:|---:|---:|
| auc_irl | 0.523 | 0.223 | +0.009 | +0.443 | +0.144 |
| auc_rf | 0.523 | 0.313 | +0.127 | +0.282 | -0.043 |
| auc_ens | 0.632 | 0.217 | +0.149 | +0.349 | -0.036 |
| gain_vs_best | 0.426 | -0.166 | +0.013 | +0.042 | +0.065 |

- **Ens AUC の説明力 R² = 0.632** (高)

## 回帰モデルから外れる repo (期待値より大きく逸脱)

### 期待値より高い AUC (positive surprise top 3)

| repo | n_review | accept_rate | actual AUC | predicted | residual |
|---|---:|---:|---:|---:|---:|
| ceilometer | 156 | 0.532 | **0.835** | 0.678 | +0.157 |
| mistral | 106 | 0.198 | **0.670** | 0.544 | +0.126 |
| horizon | 603 | 0.479 | **0.859** | 0.738 | +0.121 |

### 期待値より低い AUC (negative surprise top 3)

| repo | n_review | accept_rate | actual AUC | predicted | residual |
|---|---:|---:|---:|---:|---:|
| zaqar | 24 | 0.125 | **0.179** | 0.437 | -0.258 |
| cinder | 7151 | 0.664 | **0.762** | 0.943 | -0.181 |
| neutron | 5183 | 0.771 | **0.840** | 0.965 | -0.125 |

## 出力ファイル
- `merged_repo_metrics.csv` : basic stats + AUC 結合テーブル
- `correlation_matrix.csv` : Spearman ρ マトリクス
- `regression_summary.csv` : 線形回帰係数・R²
- `fig_corr_heatmap.pdf` : 相関ヒートマップ
- `fig_accept_vs_auc.pdf` : 承諾率 vs Ens AUC scatter
- `fig_n_review_vs_auc.pdf` : n_review vs IRL/RF/Ens の 3 panel
- `fig_repo_2d_cluster.pdf` : 2D map (n_review × accept_rate, 色 = AUC)
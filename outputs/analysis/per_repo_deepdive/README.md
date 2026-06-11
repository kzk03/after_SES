# Per-Repository Deep Dive Analysis

`outputs/analysis/per_repo/` の repo 単位集計結果を、
(1) window 別推移, (2) tier 別集計, (3) archetype 構成比, (4) cinder 深掘り
の 4 軸で深堀りした補足分析。

## 1. Window 別の repo 推移

- repo × window cell 数: 167 (auc 算出可能のみ)
### 時系列変動の大きい repo (Ens AUC std 上位 5、IRL/RF/Ens 全 window 値)

| top_dir | Ens mean | Ens std | IRL 0-3m | IRL 3-6m | IRL 6-9m | IRL 9-12m | RF 0-3m | RF 3-6m | RF 6-9m | RF 9-12m | Ens 0-3m | Ens 3-6m | Ens 6-9m | Ens 9-12m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| masakari | 0.352 | 0.486 | 0.781 | — | 0.600 | 0.000 | 0.922 | — | 0.100 | 0.000 | 0.906 | — | 0.150 | 0.000 |
| zaqar | 0.333 | 0.471 | — | — | 1.000 | 0.000 | — | — | 0.000 | 0.667 | — | — | 0.000 | 0.667 |
| vagrant | 0.750 | 0.354 | — | — | 0.500 | 0.000 | — | — | 1.000 | 0.500 | — | — | 1.000 | 0.500 |
| blazar | 0.773 | 0.321 | — | 0.545 | — | 1.000 | — | 0.523 | — | 1.000 | — | 0.545 | — | 1.000 |
| openstack_auth | 0.833 | 0.289 | 0.500 | 1.000 | — | 1.000 | 0.500 | 0.667 | — | 1.000 | 0.500 | 1.000 | — | 1.000 |

### 時系列安定な repo (Ens mean > 0.6 で std 最小 5、IRL/RF/Ens 全 window 値)

| top_dir | Ens mean | Ens std | IRL 0-3m | IRL 3-6m | IRL 6-9m | IRL 9-12m | RF 0-3m | RF 3-6m | RF 6-9m | RF 9-12m | Ens 0-3m | Ens 3-6m | Ens 6-9m | Ens 9-12m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neutron | 0.842 | 0.010 | 0.840 | 0.779 | 0.715 | 0.823 | 0.804 | 0.835 | 0.833 | 0.816 | 0.852 | 0.828 | 0.841 | 0.847 |
| doc | 0.847 | 0.013 | 0.814 | 0.799 | 0.797 | 0.804 | 0.831 | 0.836 | 0.811 | 0.829 | 0.853 | 0.848 | 0.829 | 0.857 |
| releasenotes | 0.846 | 0.017 | 0.771 | 0.817 | 0.778 | 0.799 | 0.825 | 0.834 | 0.853 | 0.864 | 0.823 | 0.858 | 0.842 | 0.861 |
| designate | 0.853 | 0.018 | 0.817 | 0.849 | 0.822 | 0.831 | 0.753 | 0.877 | 0.814 | 0.731 | 0.848 | 0.878 | 0.853 | 0.835 |
| keystone | 0.777 | 0.020 | 0.764 | 0.801 | 0.834 | 0.745 | 0.717 | 0.801 | 0.684 | 0.706 | 0.762 | 0.806 | 0.775 | 0.765 |

## 2. Tier 別の AUC 分布

| tier | n_repo | total_n_pair | median IRL | median RF | median Ens | median gain |
|---|---:|---:|---:|---:|---:|---:|
| 中 | 2 | 924 | 0.808 | 0.787 | **0.836** | +0.028 |
| 大 | 10 | 7952 | 0.792 | 0.811 | **0.830** | +0.014 |
| cross-cutting | 24 | 6512 | 0.735 | 0.711 | **0.754** | +0.001 |
| 小 | 13 | 2296 | 0.658 | 0.664 | **0.689** | -0.027 |
| 極小 | 1 | 16 | 0.733 | 0.700 | **0.600** | -0.133 |

- **Kruskal-Wallis (大/中/小/極小)**: H = 11.812, p = 0.00272 → 有意差あり

## 3. Repo × Archetype 構成比 (total_pair ≥ 50 の repo)

- 対象 repo 数: 76

### A1 (Sporadic Newcomer) 比率の高い repo top 5
| top_dir | total | A1% | A2% | A3% | A4% |
|---|---:|---:|---:|---:|---:|
| openstack_dashboard/api | 72 | **27.8** | 22.2 | 11.1 | 38.9 |
| glance/tests | 52 | **23.1** | 7.7 | 30.8 | 38.5 |
| nova/tests | 216 | **22.2** | 16.7 | 22.2 | 38.9 |
| neutron/plugins | 172 | **20.9** | 23.3 | 32.6 | 23.3 |
| manila/tests | 96 | **20.8** | 29.2 | 25.0 | 25.0 |

### A3 (Approving Veteran) 比率の高い repo top 5
| top_dir | total | A1% | A2% | A3% | A4% |
|---|---:|---:|---:|---:|---:|
| neutron/privileged | 64 | 0.0 | 18.8 | **75.0** | 6.2 |
| neutron/services | 84 | 4.8 | 9.5 | **61.9** | 23.8 |
| neutron/objects | 72 | 0.0 | 5.6 | **61.1** | 33.3 |
| neutron/api | 60 | 0.0 | 13.3 | **60.0** | 26.7 |
| tools/config | 68 | 0.0 | 0.0 | **58.8** | 41.2 |

### A2 (Strict Reviewer-Only) 比率の高い repo top 5
| top_dir | total | A1% | A2% | A3% | A4% |
|---|---:|---:|---:|---:|---:|
| playbooks/devstack | 52 | 15.4 | **38.5** | 7.7 | 38.5 |
| tacker/tests | 76 | 15.8 | **36.8** | 10.5 | 36.8 |
| manila/api | 64 | 6.2 | **31.2** | 37.5 | 25.0 |
| tacker/sol_refactored | 52 | 15.4 | **30.8** | 15.4 | 38.5 |
| ceilometer/tests | 52 | 7.7 | **30.8** | 30.8 | 30.8 |

## 4. Cinder の異常性 (RF 一貫優位) の深掘り

- sub-directory × window cell 数: 45
- Δ (=IRL−RF) 中央値: -0.064
- Δ 最大 (IRL 寄り): +0.299
- Δ 最小 (RF 寄り): -0.371

### RF が大幅優位な cinder sub-dir × window (Δ 小さい top 5)

| directory | window | n_pair | IRL | RF | Δ |
|---|---|---:|---:|---:|---:|
| cinder/image | 3-6m | 28 | 0.469 | 0.840 | -0.371 |
| cinder/db | 0-3m | 21 | 0.625 | 0.947 | -0.322 |
| cinder/api | 3-6m | 44 | 0.577 | 0.898 | -0.320 |
| cinder/db | 3-6m | 21 | 0.596 | 0.913 | -0.317 |
| cinder/zonemanager | 0-3m | 10 | 0.708 | 1.000 | -0.292 |

## 出力ファイル
- `repo_window_evolution.csv` : repo × window の生メトリクス
- `tier_summary.csv` : tier 別 AUC / gain 中央値
- `tier_stats.json` : tier 群間 Kruskal-Wallis / Mann-Whitney
- `repo_archetype_mix.csv` : repo × archetype のペア数構成比
- `cinder_breakdown.csv` : cinder sub-dir × window 詳細
- `fig_repo_window_heatmap.pdf` : repo × window の Ens AUC heatmap
- `fig_tier_boxplot.pdf` : tier 別 boxplot
- `fig_repo_archetype_stack.pdf` : repo 別 archetype 構成比 stacked bar
- `fig_cinder_breakdown.pdf` : cinder の sub-dir × window heatmap
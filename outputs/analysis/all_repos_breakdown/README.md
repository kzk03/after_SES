# 全 main repos の sub-dir × window 深掘り + 集計単位対比

Cinder に絞らず全 main repos について sub-dir × window 詳細を集計し、
project (raw review-level) と repo (pair-level) の対比を行う。

## 1. 全 main repo の Δ = IRL − RF 分布

`median_delta` 昇順 (RF 優位 ← → IRL 優位):

| repo | n_cells | n_irl_win | n_tied | n_rf_win | median Δ | min Δ | max Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| cloudkitty | 9 | 0 | 3 | 6 | -0.167 | -0.500 | +0.000 |
| tacker | 35 | 3 | 6 | 26 | -0.152 | -0.611 | +0.200 |
| venus | 10 | 2 | 2 | 6 | -0.125 | -1.000 | +0.400 |
| ceilometer | 19 | 2 | 6 | 11 | -0.083 | -0.407 | +0.083 |
| cinder | 45 | 10 | 6 | 29 | -0.064 | -0.371 | +0.299 |
| neutron | 60 | 12 | 11 | 37 | -0.059 | -0.625 | +0.219 |
| magnum | 17 | 6 | 2 | 9 | -0.033 | -1.000 | +0.750 |
| nova | 70 | 24 | 21 | 25 | -0.008 | -0.268 | +1.000 |
| swift | 23 | 10 | 5 | 8 | +0.000 | -0.800 | +0.375 |
| octavia | 36 | 11 | 11 | 14 | +0.000 | -0.333 | +0.333 |
| masakari | 3 | 1 | 1 | 1 | +0.000 | -0.500 | +0.333 |
| watcher | 3 | 1 | 1 | 1 | +0.000 | -0.500 | +0.250 |
| zaqar | 2 | 1 | 0 | 1 | +0.000 | -1.000 | +1.000 |
| horizon | 4 | 1 | 2 | 1 | +0.000 | -0.300 | +1.000 |
| heat | 12 | 4 | 6 | 2 | +0.000 | -1.000 | +0.137 |
| glance | 21 | 5 | 7 | 9 | +0.000 | -1.000 | +1.000 |
| designate | 48 | 14 | 18 | 16 | +0.000 | -1.000 | +1.000 |
| cyborg | 5 | 0 | 5 | 0 | +0.000 | +0.000 | +0.000 |
| blazar | 5 | 0 | 4 | 1 | +0.000 | -0.500 | +0.000 |
| barbican | 11 | 5 | 5 | 1 | +0.000 | -0.062 | +0.250 |
| manila | 33 | 16 | 7 | 10 | +0.015 | -0.500 | +0.443 |
| ironic | 35 | 19 | 8 | 8 | +0.036 | -0.242 | +0.500 |
| keystone | 28 | 16 | 6 | 6 | +0.042 | -0.304 | +0.688 |
| mistral | 14 | 9 | 1 | 4 | +0.077 | -0.250 | +0.500 |
| trove | 15 | 9 | 1 | 5 | +0.125 | -1.000 | +1.000 |

### 中央値での 3 群分類
- **RF 中央値優位** (median Δ < -3pp): 7 repo — cloudkitty, tacker, venus, ceilometer, cinder, neutron, magnum
- **拮抗** (|median Δ| ≤ 3pp): 14 repo — nova, swift, octavia, masakari, watcher, zaqar, horizon, heat, glance, designate, cyborg, blazar, barbican, manila
- **IRL 中央値優位** (median Δ > +3pp): 4 repo — ironic, keystone, mistral, trove

## 2. Project (raw) vs Repo (pair) 集計単位の対比

評価期間 (2022-01〜2023-01) の raw review-level と pair (window) ベースを並列表示。

| repo | proj_n_review | proj_accept | proj_n_dev | pair_n | pair_accept | pair_n_dev (n_subdir) | IRL | RF | Ens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cinder | 7151 | 0.664 | 180 | 2180 | 0.425 | (16) | 0.695 | 0.769 | **0.753** |
| neutron | 5183 | 0.771 | 120 | 1660 | 0.457 | (21) | 0.802 | 0.821 | **0.844** |
| nova | 3878 | 0.504 | 151 | 1636 | 0.443 | (20) | 0.760 | 0.799 | **0.806** |
| ironic | 1498 | 0.784 | 51 | 588 | 0.505 | (10) | 0.818 | 0.835 | **0.851** |
| octavia | 1135 | 0.584 | 73 | 696 | 0.244 | (11) | 0.862 | 0.884 | **0.888** |
| manila | 971 | 0.454 | 68 | 632 | 0.419 | (11) | 0.783 | 0.765 | **0.807** |
| tacker | 847 | 0.433 | 37 | 820 | 0.229 | (16) | 0.766 | 0.842 | **0.849** |
| horizon | 603 | 0.479 | 48 | 80 | 0.113 | (5) | 0.725 | 0.824 | **0.769** |
| swift | 524 | 0.689 | 32 | 224 | 0.562 | (7) | 0.794 | 0.822 | **0.828** |
| designate | 513 | 0.809 | 29 | 292 | 0.421 | (18) | 0.783 | 0.715 | **0.802** |
| keystone | 483 | 0.342 | 48 | 524 | 0.202 | (11) | 0.717 | 0.719 | **0.749** |
| glance | 416 | 0.514 | 31 | 216 | 0.380 | (10) | 0.751 | 0.724 | **0.765** |
| venus | 321 | 0.810 | 17 | 144 | 0.118 | (9) | 0.600 | 0.723 | **0.693** |
| heat | 278 | 0.655 | 34 | 148 | 0.372 | (5) | 0.770 | 0.787 | **0.795** |
| barbican | 204 | 0.525 | 18 | 96 | 0.292 | (4) | 0.642 | 0.633 | **0.631** |
| magnum | 183 | 0.383 | 30 | 244 | 0.291 | (7) | 0.619 | 0.539 | **0.603** |
| ceilometer | 156 | 0.532 | 27 | 288 | 0.201 | (8) | 0.816 | 0.855 | **0.862** |
| mistral | 106 | 0.198 | 17 | 300 | 0.213 | (12) | 0.616 | 0.710 | **0.703** |
| cloudkitty | 98 | 0.622 | 11 | 112 | 0.375 | (9) | 0.665 | 0.744 | **0.726** |
| trove | 87 | 0.540 | 17 | 108 | 0.380 | (9) | 0.683 | 0.669 | **0.698** |

### Project (raw) vs Pair (window) 承諾率の対応
- Spearman ρ(proj承諾率, pair承諾率) = **+0.347** (p=0.0653)
- proj_accept > pair_accept なのは pair が「window 内に依頼あり&承諾」を要求するため

## 3. IRL が大幅に劣る cell (Δ < -0.15)

- 該当 cell 数: 119
- repo 別出現数: neutron=21, tacker=18, designate=13, cinder=10, glance=8, nova=8, cloudkitty=6, venus=4, swift=3, manila=3

### Top 15 worst (IRL 大幅劣勢)

| repo | directory | window | n_pair | accept_rate | IRL | RF | Ens | Δ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| zaqar | zaqar/tests | 9-12m | 2 | 0.50 | 0.000 | 1.000 | 1.000 | -1.000 |
| glance | glance/api | 3-6m | 2 | 0.50 | 0.000 | 1.000 | 0.000 | -1.000 |
| glance | glance/policies | 0-3m | 2 | 0.50 | 0.000 | 1.000 | 0.000 | -1.000 |
| heat | heat/db | 3-6m | 3 | 0.67 | 0.000 | 1.000 | 1.000 | -1.000 |
| magnum | magnum/conductor | 6-9m | 3 | 0.67 | 0.000 | 1.000 | 0.500 | -1.000 |
| designate | designate/storage | 6-9m | 3 | 0.67 | 0.000 | 1.000 | 1.000 | -1.000 |
| designate | designate/common | 6-9m | 2 | 0.50 | 0.000 | 1.000 | 0.000 | -1.000 |
| venus | venus/conf | 6-9m | 4 | 0.25 | 0.000 | 1.000 | 1.000 | -1.000 |
| trove | trove/cmd | 9-12m | 2 | 0.50 | 0.000 | 1.000 | 0.000 | -1.000 |
| swift | swift/account | 3-6m | 6 | 0.17 | 0.200 | 1.000 | 0.600 | -0.800 |
| venus | venus | 6-9m | 6 | 0.17 | 0.000 | 0.800 | 0.800 | -0.800 |
| neutron | neutron/scheduler | 3-6m | 6 | 0.33 | 0.375 | 1.000 | 0.625 | -0.625 |
| tacker | tacker/db | 6-9m | 15 | 0.20 | 0.333 | 0.944 | 0.833 | -0.611 |
| tacker | tacker/vnflcm | 6-9m | 14 | 0.21 | 0.364 | 0.970 | 0.909 | -0.606 |
| neutron | neutron | 6-9m | 12 | 0.58 | 0.286 | 0.886 | 0.743 | -0.600 |

## 4. RF が大幅に劣る cell (Δ > +0.15)

- 該当 cell 数: 78
- repo 別出現数: designate=9, keystone=9, nova=7, manila=7, trove=6, cinder=6, mistral=5, ironic=4, glance=3, magnum=3

### Top 15 best (IRL 大幅優位)

| repo | directory | window | n_pair | accept_rate | IRL | RF | Ens | Δ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| zaqar | zaqar/tests | 6-9m | 2 | 0.50 | 1.000 | 0.000 | 0.000 | +1.000 |
| nova | nova/privsep | 9-12m | 2 | 0.50 | 1.000 | 0.000 | 1.000 | +1.000 |
| horizon | horizon/templates | 3-6m | 3 | 0.67 | 1.000 | 0.000 | 1.000 | +1.000 |
| designate | designate/common | 0-3m | 2 | 0.50 | 1.000 | 0.000 | 1.000 | +1.000 |
| designate | designate/backend | 0-3m | 3 | 0.67 | 1.000 | 0.000 | 1.000 | +1.000 |
| glance | glance/db | 3-6m | 5 | 0.80 | 1.000 | 0.000 | 0.500 | +1.000 |
| trove | trove/guestagent | 3-6m | 4 | 0.75 | 1.000 | 0.000 | 0.333 | +1.000 |
| trove | trove/guestagent | 0-3m | 4 | 0.50 | 0.750 | 0.000 | 0.000 | +0.750 |
| magnum | magnum/db | 9-12m | 4 | 0.50 | 0.750 | 0.000 | 0.250 | +0.750 |
| keystone | keystone/conf | 9-12m | 8 | 0.50 | 0.812 | 0.125 | 0.625 | +0.688 |
| glance | glance/quota | 0-3m | 4 | 0.25 | 1.000 | 0.333 | 1.000 | +0.667 |
| keystone | keystone/conf | 6-9m | 8 | 0.50 | 1.000 | 0.344 | 0.750 | +0.656 |
| nova | nova/policies | 9-12m | 6 | 0.17 | 0.600 | 0.000 | 0.600 | +0.600 |
| glance | glance/db | 9-12m | 5 | 0.40 | 1.000 | 0.417 | 1.000 | +0.583 |
| trove | trove/cmd | 0-3m | 2 | 0.50 | 1.000 | 0.500 | 1.000 | +0.500 |

## 5. 出力ファイル
- `per_repo_subdir_breakdown.csv` : 全 main repo × sub-dir × window 詳細
- `repo_delta_summary.csv` : 各 repo の Δ 分布統計
- `project_vs_repo_compare.csv` : project (raw) と repo (pair) 並列比較
- `irl_struggle_pattern.csv` : IRL が大幅劣勢の cell
- `rf_struggle_pattern.csv` : RF が大幅劣勢の cell
- `fig_all_repos_delta_heatmap.pdf` : 全 main repo × window の median Δ heatmap
- `fig_all_repos_subdir_grid.pdf` : 主要 repo の sub-dir × window grid
- `fig_project_vs_repo.pdf` : 集計単位の対比 scatter
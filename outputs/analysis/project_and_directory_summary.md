# 各 Project / 各 Directory のサマリ表

作成日: 2026-05-24
対象: OpenStack governance 配下の 32 main repos と pair_predictions 由来の全 directory
基準: 評価期間 2022-01〜2023-01、4 window pooled AUC

## 凡例

| 列名 | 意味 |
|---|---|
| n_review | 評価期間の raw レビュー件数 (combined_raw_main32.csv) |
| accept | 承諾率 = label=1 の割合 |
| n_dev | ユニーク reviewer email 数 |
| pair_n | pair_predictions の (dev, dir) ペア総数 (4 window 累計) |
| IRL / RF / Ens | 4 window pooled AUC |
| gain | Ensemble − max(IRL, RF) |
| medΔ | sub-dir × window 各 cell の (IRL − RF) の中央値 |
| IRL勝:拮:RF勝 | sub-dir × window cell の補完性 3 群分類 (Δ>+3pp / |Δ|≤3pp / Δ<-3pp) |
| — | データなし or 計算不能 |

---

## A. 各 Project (32 main repos) のサマリ

tier 順 / project 内は n_review 降順。

| tier | repo | n_review | accept | n_dev | pair_n | IRL | RF | Ens | gain | medΔ | IRL勝:拮:RF勝 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 大 | cinder | 7,151 | 0.664 | 180 | 2,180 | 0.695 | 0.769 | **0.753** | -0.016 | -0.095 | 5:7:33 |
| 大 | neutron | 5,183 | 0.771 | 120 | 1,660 | 0.802 | 0.821 | **0.844** | +0.023 | -0.045 | 13:14:33 |
| 大 | nova | 3,878 | 0.504 | 151 | 1,636 | 0.760 | 0.799 | **0.806** | +0.007 | -0.055 | 14:15:41 |
| 大 | ironic | 1,498 | 0.784 | 51 | 588 | 0.818 | 0.835 | **0.851** | +0.016 | 0.000 | 10:15:10 |
| 大 | octavia | 1,135 | 0.584 | 73 | 696 | 0.862 | 0.884 | **0.888** | +0.004 | -0.011 | 10:9:17 |
| 大 | horizon | 603 | 0.479 | 48 | 80 | 0.725 | 0.824 | **0.769** | -0.055 | -0.021 | 1:1:2 |
| 大 | swift | 524 | 0.689 | 32 | 224 | 0.794 | 0.822 | **0.828** | +0.006 | 0.000 | 7:6:10 |
| 大 | keystone | 483 | 0.342 | 48 | 524 | 0.717 | 0.719 | **0.749** | +0.029 | +0.008 | 12:5:11 |
| 大 | glance | 416 | 0.514 | 31 | 216 | 0.751 | 0.724 | **0.765** | +0.014 | 0.000 | 10:6:5 |
| 大 | heat | 278 | 0.655 | 34 | 148 | 0.770 | 0.787 | **0.795** | +0.009 | 0.000 | 2:6:4 |
| 中 | manila | 971 | 0.454 | 68 | 632 | 0.783 | 0.765 | **0.807** | +0.023 | +0.027 | 16:8:9 |
| 中 | designate | 513 | 0.809 | 29 | 292 | 0.783 | 0.715 | **0.802** | +0.019 | 0.000 | **19:16:13** |
| 小 | tacker | 847 | 0.433 | 37 | 820 | 0.766 | 0.842 | **0.849** | +0.007 | -0.054 | 7:6:22 |
| 小 | skyline-apiserver | 370 | 0.746 | 20 | — | — | — | **—** | — | — | — |
| 小 | venus | 321 | 0.810 | 17 | 144 | 0.600 | 0.723 | **0.693** | -0.029 | -0.106 | 4:0:6 |
| 小 | barbican | 204 | 0.525 | 18 | 96 | 0.642 | 0.633 | **0.631** | -0.011 | +0.125 | 7:1:3 |
| 小 | magnum | 183 | 0.383 | 30 | 244 | 0.619 | 0.539 | **0.603** | -0.016 | +0.075 | 10:0:7 |
| 小 | ceilometer | 156 | 0.532 | 27 | 288 | 0.816 | 0.855 | **0.862** | +0.006 | 0.000 | 5:7:7 |
| 小 | mistral | 106 | 0.198 | 17 | 300 | 0.616 | 0.710 | **0.703** | -0.007 | -0.031 | 5:2:7 |
| 小 | cloudkitty | 98 | 0.622 | 11 | 112 | 0.665 | 0.744 | **0.726** | -0.019 | 0.000 | 2:5:2 |
| 小 | trove | 87 | 0.540 | 17 | 108 | 0.683 | 0.669 | **0.698** | +0.015 | 0.000 | 5:3:7 |
| 小 | cyborg | 69 | 0.652 | 7 | 48 | 0.828 | 0.850 | **0.868** | +0.019 | 0.000 | 0:4:1 |
| 小 | blazar | 51 | 0.627 | 5 | 52 | 0.777 | 0.489 | **0.753** | -0.023 | 0.000 | 2:2:1 |
| 小 | masakari | 35 | 0.543 | 11 | 48 | 0.554 | 0.469 | **0.545** | -0.009 | +0.125 | 2:0:1 |
| 小 | zaqar | 24 | 0.125 | 7 | 16 | 0.214 | 0.518 | **0.250** | -0.268 | -0.500 | 0:1:1 |
| 小 | watcher | 22 | 0.455 | 14 | 20 | 0.618 | 0.392 | **0.588** | -0.029 | 0.000 | 1:1:1 |
| 小 | storlets | 9 | 0.667 | 4 | — | — | — | **—** | — | — | — |
| 極小 | freezer | 17 | 0.118 | 6 | 12 | — | — | **—** | — | — | — |
| 極小 | rally | 12 | 0.167 | 5 | — | — | — | **—** | — | — | — |
| 極小 | adjutant | 9 | 0.111 | 3 | 8 | — | — | **—** | — | — | — |
| 極小 | zun | 8 | 0.750 | 5 | 16 | 0.533 | 0.600 | **0.400** | -0.200 | — | — |
| 極小 | vitrage | 3 | 0.667 | 1 | 12 | — | — | **—** | — | — | — |

### Project 全体観察

- **大 tier 10 repo**: 全て AUC 算出可、Ens AUC レンジ 0.75 〜 0.89
- **中 tier 2 repo**: manila / designate ともに Ens AUC ≈ 0.80
- **小 tier 15 repo**: AUC 算出可は 14、レンジ 0.25 〜 0.87 (極端なばらつき)
- **極小 tier 5 repo**: AUC 算出可は 1 (zun のみ)、4 repo は pair_n 不足で評価不能

### medΔ で見る補完性パターン

- **強い RF 寄り (Δ ≤ -0.05)**: cinder, nova, neutron, tacker, venus, zaqar — **大規模 3 + 一部小規模**
- **拮抗 (|Δ| ≤ 0.03)**: 大 tier 残り + ironic, octavia, swift, glance, heat, manila, designate, ceilometer 等
- **強い IRL 寄り (Δ ≥ +0.05)**: magnum, masakari, barbican — **小規模 3**
- **特殊**: designate は medΔ=0 だが IRL 勝 19 cell (40%) で IRL が大勝ちする repo が多い

---

## B. 各 Directory のサマリ (上位 50 件、n_pair 降順)

全 directory = 390 件 (4 window 跨ぐ unique)、AUC 算出可能は 286 件。
ここでは n_pair 上位 50 件を表示。全件は `outputs/analysis/per_dir_all/directory_pooled_metrics.csv` 参照。

| # | directory | top_dir | category | n_pair | n_pos | accept | IRL | RF | Ens | gain | Δ |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | releasenotes/notes | releasenotes | cross-cutting | 1,236 | 613 | 0.50 | 0.789 | 0.846 | **0.840** | -0.006 | -0.057 |
| 2 | doc/source | doc | cross-cutting | 916 | 443 | 0.48 | 0.791 | 0.821 | **0.837** | +0.015 | -0.030 |
| 3 | cinder/volume | cinder | project-specific | 404 | 213 | 0.53 | 0.743 | 0.821 | **0.799** | -0.022 | -0.078 |
| 4 | cinder/tests | cinder | project-specific | 400 | 206 | 0.52 | 0.763 | 0.823 | **0.815** | -0.008 | -0.060 |
| 5 | api-ref/source | api-ref | cross-cutting | 388 | 153 | 0.39 | 0.692 | 0.748 | **0.743** | -0.005 | -0.056 |
| 6 | nova/tests | nova | project-specific | 300 | 119 | 0.40 | 0.834 | 0.868 | **0.878** | +0.010 | -0.034 |
| 7 | devstack | devstack | cross-cutting | 292 | 66 | 0.23 | 0.761 | 0.696 | **0.763** | +0.002 | +0.065 |
| 8 | neutron/tests | neutron | project-specific | 276 | 127 | 0.46 | 0.831 | 0.859 | **0.868** | +0.009 | -0.028 |
| 9 | neutron/plugins | neutron | project-specific | 232 | 106 | 0.46 | 0.870 | 0.906 | **0.916** | +0.011 | -0.036 |
| 10 | nova/virt | nova | project-specific | 232 | 83 | 0.36 | 0.810 | 0.868 | **0.869** | +0.001 | -0.057 |
| 11 | cinder | cinder | project-specific | 224 | 110 | 0.49 | 0.702 | 0.756 | **0.750** | -0.006 | -0.054 |
| 12 | tools | tools | cross-cutting | 212 | 38 | 0.18 | 0.782 | 0.710 | **0.797** | +0.014 | **+0.072** |
| 13 | cinder/backup | cinder | project-specific | 196 | 85 | 0.43 | 0.710 | 0.787 | **0.783** | -0.004 | -0.077 |
| 14 | devstack/lib | devstack | cross-cutting | 188 | 68 | 0.36 | 0.665 | 0.669 | **0.700** | +0.031 | -0.004 |
| 15 | zuul.d | zuul.d | cross-cutting | 180 | 111 | 0.62 | 0.829 | 0.835 | **0.874** | +0.039 | -0.006 |
| 16 | cinder/api | cinder | project-specific | 176 | 70 | 0.40 | 0.685 | 0.789 | **0.771** | -0.017 | **-0.103** |
| 17 | releasenotes/source | releasenotes | cross-cutting | 172 | 10 | 0.06 | 0.702 | 0.743 | **0.732** | -0.011 | -0.041 |
| 18 | etc | etc | cross-cutting | 168 | 31 | 0.18 | 0.850 | 0.753 | **0.861** | +0.011 | **+0.097** |
| 19 | octavia/tests | octavia | project-specific | 156 | 32 | 0.21 | 0.879 | 0.919 | **0.906** | -0.014 | -0.040 |
| 20 | nova/compute | nova | project-specific | 144 | 66 | 0.46 | 0.840 | 0.885 | **0.886** | +0.001 | -0.046 |
| 21 | neutron/agent | neutron | project-specific | 140 | 85 | 0.61 | 0.814 | 0.847 | **0.859** | +0.012 | -0.033 |
| 22 | neutron/common | neutron | project-specific | 136 | 78 | 0.57 | 0.851 | 0.893 | **0.904** | +0.010 | -0.042 |
| 23 | manila/tests | manila | project-specific | 120 | 62 | 0.52 | 0.832 | 0.823 | **0.877** | +0.044 | +0.009 |
| 24 | tacker/tests | tacker | project-specific | 116 | 45 | 0.39 | 0.672 | 0.768 | **0.755** | -0.013 | **-0.096** |
| 25 | cinder/image | cinder | project-specific | 112 | 66 | 0.59 | 0.472 | 0.642 | **0.544** | **-0.098** | **-0.170** |
| 26 | nova/conf | nova | project-specific | 112 | 52 | 0.46 | 0.761 | 0.825 | **0.817** | -0.008 | -0.064 |
| 27 | ironic/tests | ironic | project-specific | 112 | 51 | 0.46 | 0.854 | 0.806 | **0.844** | -0.010 | +0.048 |
| 28 | manila/share | manila | project-specific | 112 | 63 | 0.56 | 0.821 | 0.816 | **0.854** | +0.033 | +0.005 |
| 29 | keystone/tests | keystone | project-specific | 108 | 32 | 0.30 | 0.824 | 0.773 | **0.842** | +0.018 | +0.051 |
| 30 | neutron/db | neutron | project-specific | 104 | 57 | 0.55 | 0.792 | 0.852 | **0.844** | -0.008 | -0.060 |
| 31 | nova/network | nova | project-specific | 104 | 35 | 0.34 | 0.731 | 0.782 | **0.783** | +0.001 | -0.051 |
| 32 | octavia/amphorae | octavia | project-specific | 100 | 22 | 0.22 | 0.913 | 0.932 | **0.937** | +0.006 | -0.019 |
| 33 | nova | nova | project-specific | 100 | 47 | 0.47 | 0.753 | 0.812 | **0.816** | +0.004 | -0.059 |
| 34 | doc | doc | cross-cutting | 100 | 2 | 0.02 | 0.949 | 0.446 | **0.913** | -0.036 | **+0.503** |
| 35 | octavia/common | octavia | project-specific | 100 | 26 | 0.26 | 0.899 | 0.864 | **0.888** | -0.011 | +0.035 |
| 36 | cinder/scheduler | cinder | project-specific | 92 | 41 | 0.45 | 0.638 | 0.787 | **0.696** | -0.091 | **-0.149** |
| 37 | cinder/objects | cinder | project-specific | 92 | 9 | 0.10 | 0.532 | 0.359 | **0.497** | -0.035 | +0.173 |
| 38 | octavia/api | octavia | project-specific | 92 | 20 | 0.22 | 0.886 | 0.903 | **0.922** | +0.019 | -0.017 |
| 39 | neutron/conf | neutron | project-specific | 88 | 50 | 0.57 | 0.772 | 0.881 | **0.879** | -0.001 | **-0.109** |
| 40 | neutron/services | neutron | project-specific | 88 | 60 | 0.68 | 0.821 | 0.824 | **0.851** | +0.027 | -0.004 |
| 41 | tacker/sol_refactored | tacker | project-specific | 84 | 36 | 0.43 | 0.703 | 0.744 | **0.771** | +0.027 | -0.041 |
| 42 | nova/objects | nova | project-specific | 84 | 54 | 0.64 | 0.803 | 0.770 | **0.824** | +0.020 | +0.034 |
| 43 | ironic/drivers | ironic | project-specific | 84 | 40 | 0.48 | 0.895 | 0.872 | **0.898** | +0.003 | +0.023 |
| 44 | cinder/db | cinder | project-specific | 84 | 41 | 0.49 | 0.636 | 0.723 | **0.711** | -0.011 | -0.087 |
| 45 | ironic/common | ironic | project-specific | 80 | 45 | 0.56 | 0.844 | 0.786 | **0.829** | -0.015 | +0.058 |
| 46 | manila/api | manila | project-specific | 80 | 33 | 0.41 | 0.771 | 0.801 | **0.800** | -0.001 | -0.031 |
| 47 | nova/api | nova | project-specific | 76 | 50 | 0.66 | 0.717 | 0.836 | **0.807** | -0.029 | **-0.120** |
| 48 | openstack_dashboard/api | openstack_dashboard | cross-cutting | 76 | 17 | 0.22 | 0.718 | 0.705 | **0.710** | -0.008 | +0.013 |
| 49 | neutron/objects | neutron | project-specific | 76 | 37 | 0.49 | 0.784 | 0.913 | **0.904** | -0.009 | **-0.129** |
| 50 | nova/objects (再掲) | nova | project-specific | 76 | 48 | 0.63 | 0.682 | 0.751 | **0.738** | -0.012 | -0.069 |

### Directory 全体観察

- **最高 AUC**: octavia/amphorae (0.937), doc (0.913, 但し n_pos=2 で疑似 1.0 系), neutron/plugins (0.916)
- **最低 AUC** (n_pair ≥ 50): cinder/image (0.544, IRL=0.47!), cinder/scheduler (0.696)
- **IRL 大勝ちの注目 cell**: doc (Δ=+0.50!), etc (+0.10), tools (+0.07), devstack (+0.07), cinder/objects (+0.17)
- **RF 大勝ちの注目 cell**: cinder/image (-0.17), cinder/scheduler (-0.15), neutron/objects (-0.13), neutron/conf (-0.11), nova/api (-0.12)
- **承諾率の極端な低さ**: doc (0.02), releasenotes/source (0.06), cinder/objects (0.10) — pair window 制約で多くが弱負例

---

## C. 集計上の補足

### 32 main repos のうち pair_predictions に出現しないもの (5 repo)

- **skyline-apiserver, storlets, rally** — pair_predictions に出現せず (pair_n=—)
- 理由: 評価期間内に予測対象となる dev × dir pair が構築できないほどデータが薄い

### pair_predictions に出現するが両クラス揃わないもの (4 repo)

- **freezer, adjutant, vitrage** — pair_n は存在 (8〜12) だが両クラス n≥1 を満たさず AUC 算出不能

### 結果として AUC 算出可能 = 26 project + 24 cross-cutting top_dir = **50 集計単位**

---

## 出力ファイル

- 本サマリ md: `outputs/analysis/project_and_directory_summary.md`
- Project raw stats: `outputs/analysis/per_repo_basic_stats/project_stats.csv` (32 行)
- top_dir AUC: `outputs/analysis/per_repo/top_dir_pooled_metrics.csv` (50 行)
- 全 directory AUC: `outputs/analysis/per_dir_all/directory_pooled_metrics.csv` (390 行)
- repo 補完性分布: `outputs/analysis/all_repos_breakdown/repo_delta_summary.csv` (25 行)

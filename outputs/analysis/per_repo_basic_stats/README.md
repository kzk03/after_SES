# Directory / Repository 別の基本統計

予測精度の話に入る前に「データ特性自体が directory / repo ごとに違うのか」を見る分析。

## 1. Project (repo) 単位の基本統計 (評価期間 2022-01〜2023-01)

- 対象 project: 32 repo
- 総 review 件数: 25,264
- 総ユニーク dev 数 (重複含む): 1,127
- accept_rate 中央値: **0.542**, min=0.111, max=0.810

### accept_rate 上位/下位 5 project

| repo | n_review | accept_rate | n_dev | reviews/dev | median月次承諾dev |
|---|---:|---:|---:|---:|---:|
**上位 5 (高承諾)**

| repo | n_review | accept_rate | n_dev | reviews/dev | med月次承諾dev |
|---|---:|---:|---:|---:|---:|
| venus | 321 | **0.810** | 17 | 18.9 | 4 |
| designate | 513 | **0.809** | 29 | 17.7 | 5 |
| ironic | 1498 | **0.784** | 51 | 29.4 | 13 |
| neutron | 5183 | **0.771** | 120 | 43.2 | 28 |
| zun | 8 | **0.750** | 5 | 1.6 | 1 |

**下位 5 (低承諾)**

| repo | n_review | accept_rate | n_dev | reviews/dev | med月次承諾dev |
|---|---:|---:|---:|---:|---:|
| adjutant | 9 | **0.111** | 3 | 3.0 | 0 |
| freezer | 17 | **0.118** | 6 | 2.8 | 0 |
| zaqar | 24 | **0.125** | 7 | 3.4 | 0 |
| rally | 12 | **0.167** | 5 | 2.4 | 0 |
| mistral | 106 | **0.198** | 17 | 6.2 | 1 |

### レビュー件数上位 5 project (規模)

| repo | n_review | accept_rate | n_dev | reviews/dev | med月次承諾dev |
|---|---:|---:|---:|---:|---:|
| cinder | **7151** | 0.664 | 180 | 39.7 | 37 |
| neutron | **5183** | 0.771 | 120 | 43.2 | 28 |
| nova | **3878** | 0.504 | 151 | 25.7 | 22 |
| ironic | **1498** | 0.784 | 51 | 29.4 | 13 |
| octavia | **1135** | 0.584 | 73 | 15.5 | 10 |

### レビュー件数下位 5 project (極小規模)

| repo | n_review | accept_rate | n_dev | reviews/dev | med月次承諾dev |
|---|---:|---:|---:|---:|---:|
| rally | **12** | 0.167 | 5 | 2.4 | 0 |
| adjutant | **9** | 0.111 | 3 | 3.0 | 0 |
| storlets | **9** | 0.667 | 4 | 2.2 | 0 |
| zun | **8** | 0.750 | 5 | 1.6 | 1 |
| vitrage | **3** | 0.667 | 1 | 3.0 | 1 |

## 2. Directory 単位の基本統計

- 対象 directory: 390 (4 window 跨ぐ unique)
- 内訳: project-specific 256 / cross-cutting 134
- accept_rate 中央値: **0.200**, std=0.222
- accept_rate 範囲: min=0.000, max=1.000

### accept_rate の category 別
- **project-specific** (n=256): median accept_rate = 0.250 (IQR 0.091〜0.469)
- **cross-cutting** (n=134): median accept_rate = 0.054 (IQR 0.000〜0.203)

### accept_rate 上位/下位 directory (n_pair ≥ 50 で絞る)

**上位 5**

| directory | category | n_pair | accept_rate | n_dev |
|---|---|---:|---:|---:|
| nova/db | project-specific | 56 | **0.696** | 14 |
| neutron/services | project-specific | 88 | **0.682** | 22 |
| nova/api | project-specific | 76 | **0.658** | 19 |
| nova/objects | project-specific | 84 | **0.643** | 21 |
| test/unit | cross-cutting | 56 | **0.643** | 14 |

**下位 5**

| directory | category | n_pair | accept_rate | n_dev |
|---|---|---:|---:|---:|
| etc/cinder | cross-cutting | 64 | **0.000** | 16 |
| cinder/transfer | project-specific | 60 | **0.000** | 15 |
| skyline_apiserver/schemas | cross-cutting | 52 | **0.019** | 13 |
| doc | cross-cutting | 100 | **0.020** | 25 |
| keystone/token | project-specific | 68 | **0.029** | 17 |

## 3. Top_dir (≈ repo) 単位の集約 (pair 単位)

- 対象 top_dir: 69 (4 window 跨ぐ unique)

### サンプルが多い top_dir top 10

| top_dir | category | n_dir | n_pair | accept_rate | n_dev | pairs/dev |
|---|---|---:|---:|---:|---:|---:|
| cinder | project-specific | 16 | 2180 | 0.425 | 114 | 19.1 |
| neutron | project-specific | 21 | 1660 | 0.457 | 74 | 22.4 |
| nova | project-specific | 20 | 1636 | 0.443 | 86 | 19.0 |
| releasenotes | cross-cutting | 3 | 1412 | 0.441 | 317 | 4.5 |
| doc | cross-cutting | 7 | 1196 | 0.390 | 232 | 5.2 |
| tacker | project-specific | 16 | 820 | 0.229 | 29 | 28.3 |
| devstack | cross-cutting | 10 | 708 | 0.219 | 95 | 7.5 |
| octavia | project-specific | 11 | 696 | 0.244 | 41 | 17.0 |
| manila | project-specific | 11 | 632 | 0.419 | 37 | 17.1 |
| ironic | project-specific | 10 | 588 | 0.505 | 31 | 19.0 |

## 4. 主要観察

- **project 間の承諾率レンジ**: 0.111 (adjutant) 〜 0.810 (venus)
  → project 間で承諾文化に大きな差があり、これが予測精度の repo 依存性に寄与する可能性
- **dev 数のレンジ**: 1 〜 180
- **dev 一人あたり review 数**: 中央値 10.0, min=1.6, max=43.2
- **月次活動量の中央値レンジ**: 1 〜 610

## 出力ファイル
- `project_stats.csv` : 32 project の基本統計 (評価期間)
- `project_monthly.csv` : project × 月の活動量
- `directory_stats.csv` : directory 単位の n_pair, accept_rate, n_dev 等
- `top_dir_stats.csv` : top_dir (repo) 単位の集約
- `fig_accept_rate_dist.pdf` : 承諾率分布 (project vs directory)
- `fig_repo_monthly.pdf` : 主要 repo の月次活動量推移 (レビュー件数/承諾 dev 数)
- `fig_dev_n_vs_accept.pdf` : dev 数 vs 承諾率 scatter
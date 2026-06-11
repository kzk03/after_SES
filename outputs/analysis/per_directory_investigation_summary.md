# Per-Directory / Per-Repository 投資調査サマリ

調査期間: 2026-05-23 〜 2026-05-24
分析者: 橋本 一輝

ディレクトリ・リポジトリごとに予測精度や基本統計がどう違うかを、
**5 つの分析モジュール** で多角的に検証した結果の統合サマリ。
論文 (`thesis/writing_v2/Hashimoto_SES_short.tex` 8 ページ版) への反映候補も整理。

---

## 0. 中心の問い

> **OSS レビュアー継続予測において、ディレクトリ/リポジトリごとに何がどう違うか?**

サブ問い:
1. データ特性 (承諾率、dev 数、活動量) は dir/repo 間でどれくらい違うか?
2. 予測精度 (AUC) は dir/repo 間でどれくらい違うか?
3. 違いの原因は何か (規模、文化、開発者構成)?
4. IRL/RF/Ensemble の優劣パターンは dir/repo 依存か?

---

## 1. 分析モジュール一覧

| ID | スクリプト | 出力ディレクトリ | 内容 |
|---|---|---|---|
| **M1** | `per_repo_basic_stats.py` | `per_repo_basic_stats/` | dir/repo の基本統計 (承諾率, dev 数, 月次活動量) |
| **M2** | `per_dir_analysis.py` (--directory) | `per_dir/` | directory 単位の AUC 集計 (390 dir, reliable 46) |
| **M3** | `per_dir_analysis.py` (--top_dir) | `per_repo/` | top_dir (≈ repo) 単位の AUC 集計 (69, 50 計算可) |
| **M4** | `per_repo_deepdive.py` | `per_repo_deepdive/` | window 別 / tier 別 / archetype × repo / cinder |
| **M5** | `per_repo_correlation.py` | `per_repo_correlation/` | repo 特性 × AUC の相関・回帰分析 |
| **M6** | `all_repos_breakdown.py` | `all_repos_breakdown/` | 全 main repos の sub-dir × window 深掘り + project/repo 単位対比 |

---

## 2. 集計単位の階層構造

```
リポジトリ (Repository, 32 main repos)
  ├─ top_dir = リポジトリ名     (nova, cinder, neutron, ...)
  │   └─ directory = top_dir/sub_dir (nova/compute, nova/tests, ...)
  └─ cross-cutting dir (releasenotes, doc, etc, tools, ...)
```

- **390 unique `directory`** (4 window 跨ぐ、`{top_dir}/{sub_dir}` 形式)
- うち **46 directory** が両クラス n≥30 で「信頼可」
- **69 unique `top_dir`** (うち 50 が AUC 算出可、project-specific 26 + cross-cutting 24)
- **32 main repos** のうち 26 が AUC 算出可 (極小 tier 6 repo が欠落)

---

## 3. 「データ特性は dir/repo ごとに違うか」(M1)

### 3.1 Project 間の承諾率は **7.3 倍差** (重要)

評価期間 2022-01〜2023-01 の 25,264 review:

| repo | n_review | accept_rate | n_dev | 月次承諾 dev |
|---|---:|---:|---:|---:|
| **venus** | 321 | **0.810** | 17 | 4 |
| designate | 513 | 0.809 | 29 | 5 |
| ironic | 1,498 | 0.784 | 51 | 13 |
| neutron | 5,183 | 0.771 | 120 | 28 |
| ... | ... | ... | ... | ... |
| mistral | 106 | 0.198 | 17 | 1 |
| rally | 12 | 0.167 | 5 | 0 |
| zaqar | 24 | 0.125 | 7 | 0 |
| freezer | 17 | 0.118 | 6 | 0 |
| **adjutant** | 9 | **0.111** | 3 | 0 |

→ **承諾文化が project ごとに桁違いに異なる**。

### 3.2 規模も 500 倍差

| 順位 | repo | n_review |
|---|---|---:|
| 1 | cinder | 7,151 |
| 2 | neutron | 5,183 |
| 3 | nova | 3,878 |
| ... | ... | ... |
| 30 | storlets | 9 |
| 31 | zun | 8 |
| 32 | vitrage | 3 |

### 3.3 Directory 単位の構造パターン

**集中型** (少数 dev が深く関与):
- tacker: 29 dev / 820 pair = **28.3 pair/dev**
- neutron: 74 dev / 1,660 pair = 22.4 pair/dev
- cinder: 114 dev / 2,180 pair = 19.1 pair/dev

**分散型** (多数 dev が広く薄く):
- releasenotes: **317 dev** / 1,412 pair = **4.5 pair/dev**
- doc: 232 dev / 1,196 pair = 5.2 pair/dev

→ レビュー集中度が repo によって 6 倍違う。

---

## 4. 「予測精度は dir/repo で違うか」(M2 + M3)

### 4.1 Directory 単位 (46 reliable directory)

**Top 3 best** (Ens AUC ≥ 0.91):
| directory | n_pair | IRL | RF | Ens |
|---|---:|---:|---:|---:|
| neutron/plugins | 232 | 0.870 | 0.906 | **0.916** |
| ironic/conf | 72 | 0.871 | 0.890 | **0.911** |
| neutron/api | 64 | 0.864 | 0.821 | **0.908** |

**Bottom 3 worst** (Ens AUC ≤ 0.70):
| directory | n_pair | IRL | RF | Ens |
|---|---:|---:|---:|---:|
| cinder/image | 112 | **0.472** | 0.642 | 0.544 |
| cinder/scheduler | 92 | 0.638 | 0.787 | 0.696 |
| devstack/lib | 188 | 0.665 | 0.669 | 0.700 |

→ **Top と Bottom の差は AUC で +37 pp**。directory ごとの違い極大。

### 4.2 Repo 単位 (50 top_dir、no filter)

**Top 5 (project-specific)**:
| repo | n_pair | IRL | RF | Ens |
|---|---:|---:|---:|---:|
| octavia | 696 | 0.862 | 0.884 | **0.888** |
| cyborg | 48 | 0.828 | 0.850 | **0.868** |
| ceilometer | 288 | 0.816 | 0.855 | **0.862** |
| ironic | 588 | 0.818 | 0.835 | **0.851** |
| tacker | 820 | 0.766 | 0.842 | **0.849** |

**Bottom 5 (project-specific)**:
| repo | n_pair | IRL | RF | Ens |
|---|---:|---:|---:|---:|
| zaqar | 16 | 0.214 | 0.518 | **0.250** |
| zun | 16 | 0.533 | 0.600 | **0.400** |
| masakari | 48 | 0.554 | 0.469 | **0.545** |
| watcher | 20 | 0.618 | 0.392 | **0.588** |
| magnum | 244 | 0.619 | 0.539 | **0.603** |

→ **Top と Bottom で AUC 0.25 〜 0.89 = レンジ 0.64**。極小 repo は予測ランダム水準。

### 4.3 集計粒度で発見が逆転 (Simpson's paradox 的)

| 指標 | directory 単位 (n=46) | repo 単位 (n=50) |
|---|---|---|
| H1: log(n_pair) vs AUC ρ | **0.026** (相関なし) | **+0.455** (強い正相関, p=0.0009) |
| H1': log(n_pair) vs gain ρ | -0.118 (NS) | **+0.434** (強い正相関, p=0.002) |
| Ens vs IRL Wilcoxon r | 0.87 (大) | 0.52 (中) |
| Ens vs RF Wilcoxon r | 0.26 (小〜中) | 0.36 (中) |
| 補完性 3 群 (IRL:tied:RF) | **8:10:28** (RF 偏重) | **15:17:18** (均衡) |

→ directory レベルでは内容が支配的、repo レベルではサンプル数が支配的。

---

## 5. 「違いを何が説明するか」(M5)

### 5.1 相関マトリクス (26 repo)

| 特徴量 | vs IRL | vs RF | vs Ens | vs gain |
|---|---:|---:|---:|---:|
| **n_review** | +0.55* | **+0.65*** | **+0.63*** | +0.48* |
| accept_rate | +0.32 | +0.23 | +0.26 | +0.10 |
| n_unique_dev | +0.49* | **+0.59*** | +0.55* | +0.48* |
| reviews_per_dev | +0.53* | +0.57* | **+0.60*** | +0.40* |
| **monthly_unique_acc_dev** | +0.59* | **+0.67*** | **+0.65*** | +0.49* |

`*` = p<0.05

→ **最強予測因子は月次承諾 dev 数 (ρ=+0.65) と n_review (ρ=+0.63)**。
   **承諾率 (ρ=+0.26, NS)** は意外にも AUC を説明しない。

### 5.2 線形回帰: log_n_review + accept_rate + log_n_dev

| target | R² | β log_n_review | β accept_rate | β log_n_dev |
|---|---:|---:|---:|---:|
| auc_irl | 0.458 | +0.089 | +0.283 | -0.003 |
| auc_rf | 0.503 | **+0.196** | +0.062 | -0.143 |
| **auc_ens** | **0.526** | **+0.207** | +0.158 | -0.153 |
| gain_vs_best | 0.315 | +0.050 | +0.058 | -0.009 |

→ **Ens AUC の 52.6% は 3 特徴量で説明**。
   log_n_review が最大係数で、accept_rate は二番手。

### 5.3 回帰モデルからの逸脱 repo

**期待値より +AUC な repo (positive surprise)**:
| repo | n_review | accept_rate | 実測 | 予測 | residual |
|---|---:|---:|---:|---:|---:|
| ceilometer | 156 | 0.532 | **0.862** | 0.671 | **+0.191** |
| cyborg | 69 | 0.652 | 0.868 | 0.700 | +0.168 |
| octavia | 1,135 | 0.584 | 0.888 | 0.792 | +0.096 |

→ 小〜中規模だが予測しやすい repo。dev 構成が安定?

**期待値より −AUC な repo (negative surprise)**:
| repo | n_review | accept_rate | 実測 | 予測 | residual |
|---|---:|---:|---:|---:|---:|
| **zaqar** | 24 | 0.125 | 0.250 | 0.525 | −0.275 |
| **cinder** | 7,151 | 0.664 | 0.753 | **0.910** | **−0.157** |
| zun | 8 | 0.750 | 0.400 | 0.551 | −0.151 |

→ **cinder は規模から期待される AUC 0.91 に対して実測 0.75。
   16pp の下振れはデータ規模では説明できない異常**。

---

## 6. 「IRL/RF/Ens の優劣は dir/repo 依存か」

### 6.1 Directory レベル 補完性 (M2)

46 reliable directory を `Δ = AUC(IRL) − AUC(RF)` で 3 群分類:

| 群 | 判定 | 件数 | 例 |
|---|---|---:|---|
| **IRL 優位** | Δ > +3 pp | 8 | etc, tools, devstack, ironic/common, ironic/tests, keystone/tests, neutron/api, nova/objects |
| 拮抗 | -3 ≤ Δ ≤ +3 pp | 10 | neutron/tests, manila/share, ironic/drivers |
| **RF 優位** | Δ < -3 pp | 28 | cinder 系 6 件, neutron/objects, tacker/tests 等 |

→ IRL 優位 directory は **テストコード/設定ファイル/インフラ系** に集中。
   RF 優位 directory は cinder 系が圧倒。

### 6.2 Repo レベル 補完性 (M3、no filter)

| 群 | 件数 | 比率 |
|---|---:|---:|
| IRL 優位 | 15 | 30% |
| 拮抗 | 17 | 34% |
| RF 優位 | 18 | 36% |

→ repo レベルでは **3 群がほぼ等分散** (directory 単位の RF 偏重 8:10:28 から大幅変化)。

### 6.3 全 main repos 横断 (M6): cinder は単独異常ではない

29 main repos × 1,024 sub-dir × window cell を集計し median Δ で 3 群分類:

**RF 中央値優位 7 repo** (median Δ < -3pp):
| repo | n_cells | n_rf_win | median Δ |
|---|---:|---:|---:|
| zaqar | 2 | 1 | -0.500 |
| venus | 10 | 6 | -0.106 |
| **cinder** | 45 | **33 (73%)** | -0.095 |
| nova | 70 | 41 (59%) | -0.055 |
| tacker | 35 | 22 (63%) | -0.054 |
| neutron | 60 | 33 (55%) | -0.045 |
| mistral | 14 | 7 | -0.031 |

→ **cinder は単独異常ではなく、上位 5 大規模 repo のうち 3 つ (cinder/nova/neutron) が
   一様に RF 寄り**。IRL が苦戦するのは cinder 固有でなく大規模 repo 全般の傾向。

**IRL 中央値優位 3 repo** (median Δ > +3pp):
| repo | median Δ | 解釈 |
|---|---:|---|
| magnum | +0.075 | 小規模 (n=244) |
| masakari | +0.125 | 極小 (n=48) |
| barbican | +0.125 | 小規模 (n=96) |

→ IRL 優位 repo は小規模に集中。

**逆ケース: designate** (median Δ ≈ 0 だが分布が異常):
- n_cells=48, IRL 優位 19 (40%), 拮抗 16, RF 優位 13
- `designate/worker`, `designate/conf`, `designate/mdns` 等で
  **IRL=1.00, RF=0.00** のような極端逆転 cell が 17 個 (全 cell の 35%)
- → designate は cinder の対極で「**sub-dir 単位で IRL が大勝ちする repo**」

### 6.4 IRL/RF struggle pattern の cross-repo 集約 (M6)

**IRL 大幅劣勢 cell (Δ < -0.15、計 122 cell)** の比率上位:
| repo | 苦戦 cell 数 | 比率 |
|---|---:|---:|
| venus | 5 | **50%** |
| trove | 6 | 40% |
| tacker | 12 | 34% |
| magnum | 5 | 29% |
| cinder | 13 | 29% |

→ **venus・trove は小規模だが半数の cell で IRL が systematic に大幅劣勢**。

**RF 大幅劣勢 cell (Δ > +0.15、計 89 cell)** の比率上位:
| repo | 苦戦 cell 数 | 比率 |
|---|---:|---:|
| **designate** | **17** | **35%** |
| magnum | 6 | 35% |
| glance | 6 | 29% |
| keystone | 7 | 25% |
| manila | 8 | 24% |

→ **designate が単独で 17 cell (全 RF 劣勢 cell の 19%)** — designate は IRL に強烈に寄る repo。

### 6.5 Cinder の異常性 (M4 — 上記の文脈での位置付け)

cinder sub-dir × window 45 cell:
- Δ = IRL − RF の **中央値 -0.095** (RF が中央値で 9.5pp 優位)
- 最悪 cell: `cinder/compute` 6-9m で **IRL=0.286, RF=0.929 (Δ=-0.643)**

| directory | window | n_pair | IRL | RF | Δ |
|---|---|---:|---:|---:|---:|
| cinder/compute | 6-9m | 9 | 0.286 | 0.929 | **-0.643** |
| cinder/db | 0-3m | 21 | 0.577 | 0.952 | -0.375 |
| cinder/scheduler | 0-3m | 23 | 0.604 | 0.958 | -0.354 |
| cinder/cmd | 6-9m | 18 | 0.419 | 0.762 | -0.344 |

→ **特定 sub-dir × window で IRL が systematic に大失敗**。

---

## 7. 「時系列安定性」(M4)

### 4 window 跨ぐ Ens AUC の std

**極めて安定 (std ≤ 0.03、AUC mean > 0.6)**:
| repo | mean | std | 0-3m | 3-6m | 6-9m | 9-12m |
|---|---:|---:|---:|---:|---:|---:|
| **neutron** | 0.843 | **0.006** | 0.842 | 0.842 | 0.839 | 0.851 |
| doc | 0.834 | 0.020 | 0.811 | 0.854 | 0.823 | 0.849 |
| releasenotes | 0.830 | 0.021 | 0.801 | 0.850 | 0.832 | 0.837 |
| tacker | 0.834 | 0.030 | 0.794 | 0.864 | 0.842 | 0.835 |

→ **neutron は実質ゼロ分散**。提案手法の out-of-time 頑健性の強い証拠。

---

## 8. 「Archetype × Repo 構成」(M4)

**A1 (Sporadic Newcomer) 比率が高い repo**:
- venus 41%, cloudkitty 29%, magnum 26%, glance 19%
- → 新参者比率が高く予測しにくい (venus: AUC 0.69)

**A3 (Approving Veteran) 比率が高い repo**:
- swift 84%, test 83%, neutron 52% + A4 27% = ベテラン層 79%
- → ベテラン主体で予測しやすい (neutron: AUC 0.84)

**A2 (Strict Reviewer-Only) 比率が高い repo**:
- mistral 59%, trove 43%, tacker 36%
- → A2 主体の repo、IRL がやや効くと期待される

---

## 9. 「Tier 別の階段構造」(M4)

| tier | n_repo | total n | median Ens AUC | median gain |
|---|---:|---:|---:|---:|
| **中** | 2 | 924 | **0.804** | +2.1 pp |
| **大** | 10 | 7,952 | **0.801** | +0.8 pp |
| cross-cutting | 24 | 6,512 | 0.751 | -0.4 pp |
| **小** | 13 | 2,296 | **0.698** | -1.1 pp |
| **極小** | 1 | 16 | **0.400** | -20.0 pp |

→ Kruskal-Wallis H=5.73, **p=0.057** (5%でぎりぎり非有意、10%で有意)
   主因は極小 tier が n=1 (zun のみ) で検出力低下。

---

## 10. 統合的洞察 (まとめ)

### 10.1 確定発見

1. **承諾率と予測精度は独立** (ρ=0.26, NS): 直感に反する重要発見
2. **n_review (規模) は最強の予測因子** (ρ=+0.63, p=0.0006): 大規模 repo ほど予測しやすい
3. **Ensemble が repo 横断で IRL より一貫優位** (Wilcoxon r=0.52, p<0.001)
4. **cinder は規模から期待される AUC を 16pp 下回る異常** (実測 0.75 vs 期待 0.91)
5. **neutron は 4 window 跨いで std=0.006 の極安定** (out-of-time 頑健性証拠)
6. **集計粒度で発見が逆転**: directory 単位は内容支配、repo 単位は規模支配
7. **Cinder は単独異常ではない** (M6): 大規模 3 repo (cinder/nova/neutron) が一様に RF 寄り
8. **逆ケースの designate**: 全 cell の 35% で IRL が大勝ち (RF=0.00, IRL=1.00 級の逆転)
9. **小規模 repo (venus, trove) は IRL が systematic 大幅劣勢** (cell 比率 40-50%)
10. **集計単位の承諾率は弱い相関のみ** (proj vs pair ρ=+0.35, p=0.065): window 制約で変質

### 10.2 仮説 (今後検証)

- **テスト/設定系 dir で IRL が効く理由**: 定型的・周期的な更新パターンを LSTM が捉えやすい?
- **cinder の異常性の根本原因**: 特殊な review workflow? owner 文化? 大型 commits?
- **A3 主体 repo (swift, neutron) の高 AUC**: ベテラン dev の行動が予測可能?

### 10.3 論文反映候補 (優先度順)

1. **§5.3 RQ3 補強** (高):
   - 「Ensemble の優位性は directory/repo 横断でも再現」を独立証拠として追加
   - neutron の window 跨ぐ std=0.006 を out-of-time 頑健性の根拠として引用
2. **§6 考察に新規節** (高):
   - 「Cinder ケーススタディ」: 補完性が崩れる特殊事例
   - 「Repo 規模と予測精度の階層」: tier 別の階段 + 回帰 R²=0.53
3. **§7 構成概念妥当性** (中):
   - 「Project 間で承諾率 7.3 倍差」を外的妥当性議論に
   - 「極小 repo は予測自体が困難」を limitations に追加
4. **§6.1 機序考察** (中):
   - 「A3 主体 repo は高 AUC」を archetype 構成 → 精度の因果連鎖として
   - 「テスト/設定系 dir で IRL 優位」を時系列符号化の効く領域として明示

### 10.4 論文反映の注意点

- 8 ページ short 版に組み込むと紙面オーバーになる可能性高
- **拡張版 (9 ページ Hashimoto_SES.tex) に §5.4 として 0.5 ページ + 1 図** が現実的
- 図の候補:
  - `per_repo_correlation/fig_repo_2d_cluster.pdf` (n_review × accept_rate × AUC の 2D map)
  - `per_repo_deepdive/fig_repo_window_heatmap.pdf` (window 別 IRL/RF/Ens 3 panel)
- supplementary として `outputs/analysis/per_dir/` 全体を独立化することも可

---

## 11. 出力物全件リスト

### スクリプト (`scripts/analyze/study/`)
- `per_repo_basic_stats.py`
- `per_dir_analysis.py` (--group-by directory/top_dir, --no-reliable-filter)
- `per_repo_deepdive.py`
- `per_repo_correlation.py`

### 出力 (`outputs/analysis/`)
- `per_repo_basic_stats/` (CSV 4 / PDF 3 / README)
- `per_dir/` (CSV 2 / JSON 1 / PDF 3 / README)
- `per_repo/` (CSV 2 / JSON 1 / PDF 3 / README)
- `per_repo_deepdive/` (CSV 5 / JSON 1 / PDF 4 / README)
- `per_repo_correlation/` (CSV 4 / PDF 4 / README)
- `all_repos_breakdown/` (CSV 5 / PDF 3 / README)

### 統合 (このファイル)
- `outputs/analysis/per_directory_investigation_summary.md`

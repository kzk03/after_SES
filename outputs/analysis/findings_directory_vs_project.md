# OSS レビュアー継続予測: ディレクトリ別・プロジェクト別の傾向の違い

作成日: 2026-05-24
作成者: 橋本 一輝
対象: MCE-IRL (LSTM) + Random Forest + Uniform Ensemble の予測精度

---

## 結論 (TL;DR)

**ディレクトリ単位・プロジェクト単位ともに予測精度の傾向は明確に違う (YES)**。

両単位で傾向は異なるが、**違いの構造** が異なる:
- **Directory 単位** → サンプル数 + **内容** で違いが生まれる
- **Project 単位** → サンプル数 (規模) + **文化** で違いが生まれる

両単位とも **大規模ほど予測しやすく、小規模で予測ランダム水準** という共通構造があるが、
**Directory 単位ではテスト/設定系で IRL 優位 / cinder 系で RF 優位** という意味的クラスタが現れ、
**Project 単位では大規模 3 repo (cinder/nova/neutron) が一様に RF 寄り、小規模で IRL 寄り** が点在する。

---

## 用語整理

| 用語 | 意味 | 件数 |
|---|---|---|
| **Project** | `data/service_teams_main_repos.csv` の main_repo (例: openstack/nova) | **32** |
| **top_dir** | pair_predictions の `directory` 列の先頭セグメント (例: nova) | 69 unique (50 が AUC 算出可) |
| **directory** | pair_predictions の `directory` 列 (例: nova/compute) | 390 unique (286 が AUC 算出可) |

**top_dir = 50 の内訳**:
- project-specific 26 (main_repos と一致、6 つの極小 repo は AUC 不能)
- cross-cutting 24 (releasenotes, doc, etc, tools 等の repo 横断 dir)

---

## Directory 単位の違い (n=286)

集計対象: 4 window 跨ぐ unique directory 390 のうち、両クラス >=1 で AUC 算出可能な **286 directory**。

### 違いの構造

| 観点 | 値 | 解釈 |
|---|---|---|
| Ens AUC レンジ | 0.00 〜 1.00 | 小サンプル dir 含むため両端に振れる |
| Ens AUC IQR | 0.62 〜 0.85 | 中央 50% は 0.62-0.85 に集中 |
| サンプル数との相関 | **Spearman ρ = +0.30 (p=2e-7)** | 大きい directory ほど予測しやすい |
| gain との相関 | **ρ = +0.33 (p=1e-8)** | 大きい directory ほど Ensemble が効く |
| 補完性 (Δ = IRL−RF) | IRL 偏 / 拮抗 / RF 偏 が混在 | 内容次第で傾向が逆転 |

### 重要発見 (Directory 単位)

1. **project-specific vs cross-cutting で有意差** (Mann-Whitney p=0.012):
   - project-specific (209 dir): median Ens AUC = **0.783**
   - cross-cutting (77 dir): median Ens AUC = **0.690** (-9.3 pp)
2. **Ensemble は IRL に対して安定優位** (Wilcoxon r=0.43, p=4e-11)
3. **Ensemble は RF にもわずかに優位** (Wilcoxon r=0.11, p=0.014)
4. **テストコード/設定/インフラ系で IRL 優位**: `etc`, `tools`, `devstack`, `ironic/tests`, `neutron/api` 等
5. **cinder 系 sub-dir で RF 優位**: `cinder/image`, `cinder/scheduler`, `cinder/db` 等で −15 〜 −17 pp の RF 圧勝

---

## Project (repo) 単位の違い (n=32, うち 26 が AUC 算出可)

集計対象: main_repos.csv の **32 project**。うち極小 tier の 6 repo (adjutant, freezer, rally, skyline-apiserver, storlets, vitrage) は両クラス揃わず除外、残り **26 project + 24 cross-cutting top_dir = 50 集計単位**。

### 違いの構造

| 観点 | 値 | 解釈 |
|---|---|---|
| Ens AUC レンジ | **0.25 〜 0.89** (差 0.64) | Directory より広いレンジ |
| 承諾率レンジ | **0.111 (adjutant) 〜 0.810 (venus)** | **7.3 倍差** |
| 規模レンジ | **3 件 〜 7,151 件** | **約 2,400 倍差** |
| サンプル数との相関 | **Spearman ρ = +0.45 (p=0.0009)** | 規模で AUC が説明できる |
| 線形回帰 R² | **0.526** | 3 特徴量で AUC の半分を説明 |

### 重要発見 (Project 単位)

1. **承諾率と AUC は独立** (ρ=0.26, NS) — 直感に反する重要発見
2. **規模 (n_review) が最強の予測因子** (ρ=+0.63, p<0.001)
3. **大規模 3 repo (cinder, nova, neutron) が一様に RF 寄り**:
   - cinder: median Δ = −0.095 (33/45 cell で RF 優位)
   - nova: −0.055 (41/70 で RF 優位)
   - neutron: −0.045 (33/60 で RF 優位)
4. **小規模 3 repo (magnum, masakari, barbican) で IRL 中央値優位**
5. **逆ケース: designate** — 同規模 (n_cells=48) だが IRL 大勝ち 19 cell (40%)
6. **neutron の時系列安定性**: 4 window 跨いで Ens AUC std=0.006 (out-of-time 頑健性証拠)
7. **回帰モデルから逸脱**:
   - **cinder**: 期待 AUC 0.910 vs 実測 0.753 (**−16 pp 下振れ**)
   - **ceilometer**: 期待 0.671 vs 実測 **0.862 (+19 pp 上振れ)**

---

## 両単位の対比

| 観点 | Directory 単位 (286) | Project 単位 (32 / 50 top_dir) |
|---|---|---|
| **集計対象** | 286 directory (例: nova/compute) | 26 project + 24 cross-cutting top_dir |
| **AUC レンジ** | 0.00 〜 1.00 (小サンプル含む) | **0.25 〜 0.89** |
| **AUC IQR** | 0.62 〜 0.85 | より広い |
| **サンプル数依存 ρ** | +0.30 (p=2e-7) | **+0.45 (p=0.0009)** |
| **承諾率レンジ** | 0.00 〜 1.00 (dir 単位) | 0.11 〜 0.81 (project 単位) |
| **承諾率 → AUC** | (未計算) | ρ=0.26 (**独立**) |
| **規模レンジ** | n_pair 1 〜 1,236 | n_review 3 〜 7,151 |
| **Ens vs IRL r** | 0.43 (中) | 0.52 (中) |
| **Ens vs RF r** | 0.11 (小) | 0.36 (中) |
| **補完性の見え方** | テスト/設定系で IRL 優位、cinder 系で RF 優位 | 大規模で RF 寄り、小規模で IRL 寄りが点在 |
| **違いを決める軸** | サンプル数 + **内容** | サンプル数 + **規模 + 文化** |

---

## 構造的観察: cinder vs designate (同規模、正反対)

| 指標 | cinder | designate |
|---|---|---|
| n_cells (sub-dir × window) | 45 | 48 |
| **IRL 優位 cell** (Δ > +3pp) | 5 (11%) | **19 (40%)** |
| 拮抗 cell | 7 (16%) | 16 (33%) |
| **RF 優位 cell** (Δ < −3pp) | **33 (73%)** | 13 (27%) |
| median Δ | −0.095 | +0.000 |
| 極端な逆転例 | `cinder/compute` で IRL=0.29, RF=0.93 | `designate/worker` で IRL=1.00, RF=0.00 |

→ 同規模 (n_cells ≈ 同) の repo でも **補完性が真逆** に現れる。

---

## 全 main repos 横断の補完性スペクトラム

29 main repos × 1,024 sub-dir × window cell を median Δ で 3 群分類:

```
RF 寄り (7 repo)          拮抗 (15 repo)              IRL 寄り (3 repo)
─────────────             ─────────────              ─────────────
zaqar    Δ=-0.50          horizon, octavia,          magnum   Δ=+0.08
venus    -0.11            ceilometer, swift,         masakari +0.13
cinder   -0.10            watcher, ironic, heat,     barbican +0.13
nova     -0.06            glance, designate,
tacker   -0.05            cyborg, cloudkitty,
neutron  -0.05            blazar, trove,
mistral  -0.03            keystone, manila
```

→ **大規模 repo は RF 寄り、小規模で IRL 寄り、中規模が拮抗** という階段構造。

---

## 集計単位の対比 (Project raw vs top_dir pair)

| repo | proj_n_review | proj_accept | pair_n | pair_accept | Ens AUC |
|---|---:|---:|---:|---:|---:|
| cinder | 7,151 | 0.664 | 2,180 | 0.425 | 0.753 |
| neutron | 5,183 | 0.771 | 1,660 | 0.457 | 0.844 |
| nova | 3,878 | 0.504 | 1,636 | 0.443 | 0.806 |
| ironic | 1,498 | 0.784 | 588 | 0.505 | 0.851 |
| **venus** | 321 | **0.810** | 144 | **0.118** | 0.693 |

- proj_accept と pair_accept は **Spearman ρ = +0.35 (p=0.065)** — 弱い相関のみ
- venus: raw 0.81 → pair 0.12 (大幅低下) — raw 高承諾だが window 内依頼が限られる
- **集計単位を変えると承諾率は別物になる**

---

## 統計的サマリ

| 検証 | 結果 | 結論 |
|---|---|---|
| Directory 単位 ρ(log n, Ens AUC) | +0.30 (p=2e-7) | **有意な正相関** |
| Project 単位 ρ(log n, Ens AUC) | +0.45 (p=0.0009) | **より強い正相関** |
| Directory 単位 Ens vs IRL Wilcoxon | r=0.43, p=4e-11 | **大効果量で IRL 上回る** |
| Project 単位 Ens vs IRL Wilcoxon | r=0.52, p<0.001 | **大効果量で IRL 上回る** |
| Directory 単位 project vs cross-cutting | U=9603, **p=0.012** | **有意差あり (-9.3 pp)** |
| 全 main repo の median Δ 分類 | RF 7 / 拮抗 15 / IRL 3 | **規模に応じた補完性スペクトラム** |
| IRL 大幅劣勢 cell | 122 個 (29 repo) | 偏在する局所異常 |
| RF 大幅劣勢 cell | 89 個 (designate が 17 で突出) | designate 集中 |

---

## 最終 take-away (10 件)

1. **ディレクトリ別 / プロジェクト別 ともに傾向は明確に違う**
2. **両単位ともサンプル数が AUC を説明する主因子** (ρ=+0.30 / +0.45)
3. **承諾率と AUC は独立** (ρ=0.26, NS) — 直感に反する
4. **Ensemble が両単位で IRL に一貫優位** (r=0.43-0.52, p<0.001)
5. **大規模 3 repo (cinder/nova/neutron) は一様に RF 寄り** = cinder は単独異常ではない
6. **小規模 3 repo (magnum/masakari/barbican) で IRL 中央値優位**
7. **designate は IRL 大勝ち 17 cell (35%)** — cinder の対極
8. **テスト/設定/インフラ系 directory で IRL 優位** (etc, tools, devstack 等)
9. **neutron は 4 window 跨いで std=0.006 の極安定** (out-of-time 頑健性証拠)
10. **集計単位の承諾率は弱い相関のみ** (proj vs pair ρ=+0.35) — window 制約で変質

---

## 論文反映候補 (優先度順)

| 優先度 | 反映先 | 内容 |
|---|---|---|
| **高** | §5.3 RQ3 補強 | Ens の優位性は directory/project 両単位で再現 (両方 p<0.001) |
| **高** | §6 考察に新規節 | 補完性スペクトラム: 大規模 repo は RF 寄り, 小規模で IRL 寄り |
| 中 | §7 構成概念妥当性 | 承諾率 7.3 倍差を外的妥当性議論に |
| 中 | §6.1 機序考察 | テスト/設定系 dir で IRL 優位 = 時系列符号化の効く領域 |
| 低 | §5.4 (拡張版のみ) | 0.5 ページ + 1 図 (fig_repo_2d_cluster.pdf or fig_all_repos_delta_heatmap.pdf) |

---

## 全分析モジュール一覧

| ID | スクリプト | 出力 | 内容 |
|---|---|---|---|
| M1 | `per_repo_basic_stats.py` | `per_repo_basic_stats/` | 承諾率, dev 数, 月次活動量 |
| M2 | `per_dir_analysis.py --group-by directory` | `per_dir/` (46 reliable), `per_dir_all/` (286 全) | directory 単位 AUC |
| M3 | `per_dir_analysis.py --group-by top_dir` | `per_repo/` | top_dir (= 26 project + 24 cross-cutting) 単位 AUC |
| M4 | `per_repo_deepdive.py` | `per_repo_deepdive/` | window 別 / tier 別 / archetype × repo |
| M5 | `per_repo_correlation.py` | `per_repo_correlation/` | 特性 × AUC の相関・回帰 |
| M6 | `all_repos_breakdown.py` | `all_repos_breakdown/` | 全 main repos の sub-dir × window |

合計: スクリプト 5 件、出力ディレクトリ 6 件、CSV 約 20 件、PDF 約 20 件、README 6 件。

---

## 関連ファイル

- 詳細統合 md: `outputs/analysis/per_directory_investigation_summary.md`
- 各モジュール README: `outputs/analysis/per_*/README.md`, `all_repos_breakdown/README.md`
- 本ファイル (clean summary): `outputs/analysis/findings_directory_vs_project.md`

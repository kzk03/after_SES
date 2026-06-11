# Directory-level Prediction Trend Analysis

OSS レビュアー継続予測 (MCE-IRL + RF + Uniform Ensemble) を **directory (モジュール) 単位**
で集計し、予測精度の傾向に違いがあるかを多角的に検証した。

## 実行

```bash
uv run python scripts/analyze/study/per_dir_analysis.py
```

## データ概要

- 入力: `outputs/main32_mce_volat_ep150/lstm_hybrid_snapshot/train_W/eval_W/pair_predictions.csv`
  (W ∈ {0-3m, 3-6m, 6-9m, 9-12m})
- ディレクトリ (`directory` 列の unique 値, 例: `nova/compute`, `releasenotes/notes`): **390 件**
- うち **auc_reliable (n_pos ≥ 30 かつ n_neg ≥ 30): 46 件** に解釈を限定
- 内訳: project-specific (top_dir が 32 main repos の一つと一致) **36 件** / cross-cutting **10 件**

`directory` 列は `{top_dir}/{sub_dir}` の 2 階層パス。
`top_dir` が `neutron`, `nova`, `cinder` 等の repo 名であれば project-specific、
`releasenotes`, `doc`, `etc`, `tools`, `devstack` 等の cross-repo な共通 dir であれば cross-cutting。

## 主要発見

### 結論: 「ディレクトリごとの傾向は違うか?」→ **Yes、明確に違う**

1. **Ensemble が directory 横断で IRL に一貫優位 (大効果量)**
2. **特定 project (cinder) で IRL が systematically 苦戦**
3. **テスト/設定/インフラ系 directory で IRL が RF を上回る局所領域あり**
4. **サンプル数と AUC は無相関**（データ量だけでは予測しやすさは決まらない）

### H1: サンプル数と AUC の相関 — 棄却 (相関なし)

- Spearman ρ(log n_pair, Ens AUC) = **0.026** (p = 0.866) → **ほぼ相関なし**
- 解釈: directory の "予測しやすさ" は単純なサンプル量では説明できない。
  内容そのもの (どんなレビュー活動か) が支配的因子。
- 構成概念妥当性 §7 への補強材料。

### H2: project-specific vs cross-cutting — 傾向はあるが有意ではない

- Mann-Whitney U = 241, p = 0.107
- 中央値: project-specific **0.835** vs cross-cutting **0.785** (+5.0 pp)
- 判定: 数値差はあるが統計的に有意ではない (n の偏り 36 vs 10 で検出力低)
- 解釈: project-specific directory の方が予測しやすい傾向はあるが、
  cross-cutting (releasenotes, doc 等) でも AUC 0.78 以上を保つため大差にはならない。

### H3: Ensemble の単独優位性 — **強く支持 (Yes)**

| 比較 | W | p_bonferroni | r (rank-biserial) | 解釈 |
|---|---:|---:|---:|---|
| Ens vs IRL | 18 | **1.4e-11** | **0.870** | 大効果量、46 中ほぼ全 directory で Ens > IRL |
| Ens vs RF  | 306 | 0.019 | 0.261 | 小〜中効果量、過半数で Ens > RF |

- 主結果: **Ensemble は IRL に対しては圧倒的、RF に対しては緩やかに上回る**。
- これは論文 §5.3 (RQ3) の "ペア単位 +2.8 pp" の主張を **directory 集計でも独立に再現**する。

### H4 (post-hoc): 補完性パターンの project 依存

**46 reliable directory を Δ = AUC(IRL) − AUC(RF) の符号で 3 群分類**:

| 群 | 判定基準 | 件数 | 比率 |
|---|---|---:|---:|
| IRL 優位 | Δ > +3 pp | **8** | 17% |
| 拮抗 | −3 ≤ Δ ≤ +3 pp | **10** | 22% |
| RF 優位 | Δ < −3 pp | **28** | 61% |

→ **RF 優位が IRL 優位の 3.5 倍**。具体的パターン:

#### IRL 優位 (Δ = IRL−RF > +3 pp) — 8 件
| directory | top_dir | n_pair | IRL | RF | Δ |
|---|---|---:|---:|---:|---:|
| etc | etc (cross-cutting) | 168 | 0.850 | 0.753 | +9.7 |
| tools | tools (cross-cutting) | 212 | 0.782 | 0.710 | +7.2 |
| devstack | devstack (cross-cutting) | 292 | 0.761 | 0.696 | +6.5 |
| ironic/common | ironic | 80 | 0.844 | 0.786 | +5.8 |
| keystone/tests | keystone | 108 | 0.824 | 0.773 | +5.1 |
| ironic/tests | ironic | 112 | 0.854 | 0.806 | +4.8 |
| neutron/api | neutron | 64 | 0.864 | 0.821 | +4.3 |
| nova/objects | nova | 84 | 0.803 | 0.770 | +3.4 |

→ **テストコード、設定ファイル、インフラ系 directory で IRL が優位**。
  これらは定型的・周期的な更新パターンを持つため時系列符号化が効きやすい。

#### RF 優位 (Δ = IRL−RF < −3 pp) — 28 件 (cinder 多数)
| directory | top_dir | n_pair | IRL | RF | Δ |
|---|---|---:|---:|---:|---:|
| cinder/image | cinder | 112 | 0.472 | 0.642 | **−17.0** |
| cinder/scheduler | cinder | 92 | 0.638 | 0.787 | **−14.9** |
| neutron/objects | neutron | 76 | 0.784 | 0.913 | −12.9 |
| neutron/conf | neutron | 88 | 0.772 | 0.881 | −10.9 |
| cinder/api | cinder | 176 | 0.685 | 0.789 | −10.3 |
| tacker/tests | tacker | 116 | 0.672 | 0.768 | −9.6 |
| cinder/db | cinder | 84 | 0.636 | 0.723 | −8.7 |
| cinder/volume | cinder | 404 | 0.743 | 0.821 | −7.8 |
| cinder/backup | cinder | 196 | 0.710 | 0.787 | −7.7 |

→ **cinder という単一 repo が RF 優位の半数を占める異常事象**。

### Top_dir 別の Ens AUC 中央値 (主要 repos)

| top_dir | n_dir (reliable) | median Ens AUC | median gain |
|---|---:|---:|---:|
| octavia | 1 | **0.906** | −0.014 |
| neutron | 9 | 0.879 | +0.010 |
| ironic | 4 | 0.871 | −0.003 |
| keystone | 1 | 0.842 | +0.018 |
| releasenotes | 1 | 0.840 | −0.006 |
| doc | 1 | 0.837 | +0.015 |
| nova | 8 | 0.820 | +0.002 |
| tools | 1 | 0.797 | +0.014 |
| **cinder** | 8 | **0.760** | **−0.014** |
| devstack | 2 | 0.732 | +0.016 |

→ **cinder の median AUC が他より明確に低い (0.760)** かつ ensemble gain も負。
  これは「cinder では IRL の追加が逆効果を生むケースが多い」=
  RF の累積統計が他 repo より強い情報を持つ可能性。

## 含意

### 論文への反映候補 (今後判断)

- **§5.3 (RQ3) の補強**: "Ensemble の優位性は directory 横断でも再現" を
  Wilcoxon r=0.87 (vs IRL) で支持できる
- **§6 考察への追加**: "cinder という特定 project で IRL が苦戦する" を新しい補完性パターンとして
  記述可能 (現状 archetype 軸でしか論じられていない project 依存の側面を補強)
- **§7 構成概念妥当性**: H1 の "サンプル数と AUC は無相関" は外的妥当性議論に活用可能

### 残された課題 (今回スコープ外)

- archetype × directory の交互作用 (n_pair が薄くなり統計困難)
- cinder が systematically RF 優位な理由の深掘り (cinder の review pattern の特殊性?)
- IRL 優位の "テスト/設定系" 仮説のさらなる検証

## 出力ファイル

- `dir_window_metrics.csv` — directory × window × {IRL, RF, Ens} の AUC, n_pair, n_pos, n_neg
- `dir_pooled_metrics.csv` — directory 単位 4 window pool + 1000 iter bootstrap CI + top_dir + category
- `stats_test.json` — Spearman / Mann-Whitney / paired Wilcoxon の検定結果
- `fig_dir_auc_ranked.pdf` — Top/Bottom 10 directory の AUC ranked bar (IRL/RF/Ens 三系列, Ens の 95% CI)
- `fig_category_violin.pdf` — project-specific vs cross-cutting の Ens AUC 分布
- `fig_n_vs_auc_scatter.pdf` — log10(n_pair) vs Ens AUC の scatter (Spearman 表示)

## 解釈方針

- `auc_reliable=False` の directory (n_pos<30 or n_neg<30) は信頼区間が広く個別解釈は避ける
- 主要解釈は **reliable な 46 directory** に限定
- archetype × directory の交互作用は今回スコープ外

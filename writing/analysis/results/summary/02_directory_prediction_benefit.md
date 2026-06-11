# ディレクトリ単位で予測するメリット

論文の主張: **「開発者単位で『この人は今後も活動するか』を 1 つの数値で予測する」のではなく、(開発者 × ディレクトリ) ペア単位で予測することの構造的意義**を示す。期間別予測のメリット (`01_horizon_prediction_benefit.md`) と相互補完。

---

## 1. ディレクトリ単位で予測性が大きく異なる

### 図: 全 46 ディレクトリ AUC ランキング

![all dirs ranked](../fig_all_dirs_ranked.png)

**読み方**: 各行 = 1 dir、横軸 = AUC (4 horizon pooled)、3 系列 (IRL / RF / Ens) を並べて表示。各 dir の右に n_pair を注記。

**ポイント**: ディレクトリごとに **AUC 0.62 から 0.92 まで 30pp 以上の幅** がある = 「予測しやすい dir」と「予測しにくい dir」が明確に分かれる。開発者単位の集約では失われる情報。

### 主要な数値結果

全 46 ディレクトリ (ペア数≥30 で安定推定可能):

| 指標 | mean | median | min | max |
|---|---:|---:|---:|---:|
| IRL AUC | 0.776 | 0.793 | 0.556 | 0.917 |
| RF AUC | 0.806 | 0.811 | 0.653 | 0.922 |
| **Ens AUC** | **0.820** | **0.825** | 0.615 | 0.919 |

予測しやすい dir TOP5: `octavia/tests` (0.93), `nova/compute` (0.92), `neutron/plugins` (0.90), `nova/virt` (0.88), `zuul.d` (0.88)
予測しにくい dir BOTTOM5: `tools` (0.71), `devstack` (0.71), `api-ref/source` (0.73), `devstack/lib` (0.75), `cinder/api` (0.76)

---

## 2. ディレクトリ別の IRL vs RF 補完性

### 図: dir 別 IRL AUC vs RF AUC 散布図

![dir auc irl vs rf](../fig_dir_auc_irl_vs_rf.png)

**読み方**: 各点 = 1 dir、x 軸 = RF AUC, y 軸 = IRL AUC。対角線より上 → IRL 優位、下 → RF 優位。点のサイズ = ペア数、色 = IRL − RF 差 (青 = IRL 勝ち、赤 = RF 勝ち)。

**ポイント**: 対角線下に多くの点 = RF が広く優位 (19 / 23 dir)。ただし**「どの dir で IRL が勝つか」**には系統的な構造がある (横断・基盤系 vs 専門特化系)。

### 図: Δ = IRL − RF 全 46 dir ランキング

![all dirs delta](../fig_all_dirs_delta_irl_rf.png)

**読み方**: 青 = IRL 優位 (上位)、赤 = RF 優位 (下位)、灰色帯 |Δ|<0.03 が拮抗領域。

### 主要な数値結果 (Δ = IRL − RF 別の分類)

| 区分 | dir 数 | 例 |
|---|---:|---|
| IRL 大勝ち (Δ > +0.05) | 4 | etc, ironic/common, nova/objects, ironic/tests |
| IRL 優位 (0 < Δ ≤ +0.05) | 11 | doc, devstack, devstack/lib, tools (横断系) |
| 拮抗 (\|Δ\| ≤ 0.03) | 13 | octavia/tests, neutron/conf, neutron/agent (安定領域) |
| RF 優位 (−0.05 ≤ Δ < 0) | 11 | nova/compute, neutron/plugins (大規模専門系) |
| **RF 大勝ち (Δ < −0.05)** | **20** | cinder/api, cinder/db, cinder/image (専門特化系) |

---

## 3. プロジェクト内の下位ディレクトリ別ばらつき (cinder の例)

### 図: cinder の下位 8 ディレクトリ

![cinder breakdown](../fig_dirs_by_project_cinder.png)

**読み方**: cinder 配下の下位 dir 別 AUC + Δ。

**ポイント**: 同じ cinder プロジェクト内でも:
- `cinder/tests` Ens AUC=0.82 (中程度予測可能、RF 優位)
- `cinder/api` Ens AUC=0.76 (予測しにくい、RF 大勝ち Δ=−0.15)
- `cinder/image` Ens AUC=0.62 (最も予測しにくい)

→ **プロジェクト単位で集約すると消える「サブ領域ごとの予測性の違い」**が dir 単位なら見える。

### 図: neutron の下位 9 ディレクトリ (対照例)

![neutron breakdown](../fig_dirs_by_project_neutron.png)

**ポイント**: neutron は AUC レンジが狭く (0.85-0.92) 全体的に予測しやすい。ただし `neutron/objects` で RF が大勝ち、`neutron/tests` で拮抗、と内部にも違いがある。

---

## 4. ディレクトリ × アーキタイプ構成の関連

### 図: dir × archetype 構成比 (top/bottom 10)

![dir archetype stacked](../fig_dir_archetype_stacked.png)

**読み方**: 上段 = 予測しやすい dir TOP10、下段 = 予測しにくい dir BOTTOM10。各 dir で開発者アーキタイプ (A1-A4) とペアアーキタイプ (P1-P4) の比率を積み上げ棒で表示。

**ポイント**: 予測しにくい dir は **A4 常時活動型 + P1 ディレクトリ未経験**の組合せに集中 = 「忙しい人が馴染みのない領域に呼ばれる」構造。一方、予測しやすい dir は archetype が多様に混在。

### 主要な数値結果 (TOP10 vs BOTTOM10 の archetype 比率差)

| archetype | TOP10 平均 | BOTTOM10 平均 | 差 | 解釈 |
|---|---:|---:|---:|---|
| A4 常時活動型 | 0.352 | 0.488 | **−0.14** | **予測しにくい dir に集中** |
| A1 低活動の新人 | 0.133 | 0.072 | +0.06 | 予測しやすい dir に多い |
| P1 ディレクトリ未経験 | 0.241 | 0.315 | **−0.07** | **予測しにくい dir に多い** |
| P4 中程度関与 | 0.288 | 0.221 | +0.07 | 予測しやすい dir に多い |

→ **ディレクトリ単位の予測性は archetype 構成と整合**: 「どんな開発者が呼ばれる dir か」が「予測しやすいか」を決める。

---

## 5. ディレクトリ規模が予測性の最強因子

### 図: dir メタ回帰

![dir meta regression](../fig_dir_meta_regression.png)

**読み方**: 左 = 安定推定可能 46 dir (OLS), 中 = 全 286 dir (sqrt(n) 重み WLS), 右 = log(n) の partial 効果図。

**ポイント**: log10(ペア数) が AUC の最強因子 (WLS 係数 +0.085, p<0.001)。承諾率も独立に正の効果 (+0.039, p<0.001)。

### 主要な数値結果 (全 286 dir WLS, 標準化後)

| 変数 | 係数 | p | 解釈 |
|---|---:|---:|---|
| log10(ペア数) | **+0.085** | <0.001 | サンプル規模 1σ↑ で AUC +0.085 |
| 承諾率 | **+0.039** | <0.001 | 承諾率 1σ↑ で AUC +0.039 |
| 関与開発者数 | −0.029 | 0.004 | 規模統制後はわずかに難化 |
| 開発者 A4 比率 | +0.020 | 0.055 | 限界的に正 |
| 基盤系フラグ | −0.057 | 0.072 | devstack/tools/etc 系は限界的に難 |

R² = 0.227 (全 286 dir)。ただし安定推定可能 46 dir に絞ると R²=0.034 → **「ペア数≥30」を超えると規模の効果は飽和**。

---

## 6. ディレクトリ単位での危険セル予測 (期間別予測との交差)

期間別予測 (`01_horizon_prediction_benefit.md`) との重要な交差: **同じ dir でも遷移ごとに予測しやすさが大きく異なる**。

例: `devstack`
- 3-6m → 6-9m attrition 予測: IRL AUC = **0.86** (RF 0.56) — IRL が圧倒
- 6-9m → 9-12m attrition 予測: Ens AUC = **0.12** (ランダム以下、完全失敗)

→ **「devstack は予測しにくい」とディレクトリ単位で言うのは粗い**。期間 × ディレクトリの組合せで予測しやすさが大きく変動する。

---

## まとめ: ディレクトリ単位で予測するメリット (4 点)

1. **dir 間で AUC 0.62-0.92 の幅**: 開発者単位の集約では失われる「どの領域で予測しやすいか」の構造が見える
2. **dir 内のサブ領域差**: 同じ project でも下位 dir で AUC が 0.62-0.82 と変動 (cinder の例)
3. **IRL/RF の役割分担が dir 別に明確**: 横断・基盤系 = IRL、専門特化系 = RF。論文の補完性主張の具体的根拠
4. **規模 + 承諾率 + archetype 構成で説明可能**: dir 単位なら「なぜ予測しやすいか」を定量的に因子分解できる

→ 論文の「**なぜ (開発者 × ディレクトリ) ペア単位で予測するか**」の根拠として、この 4 点を §3.2 や §5.2 で主張可能。

---

## 出典分析

- §3 ディレクトリ別予測性 → `scripts/03_directory_predictability.py`
- §5 thesis project/repo 別深堀り → `writing/analysis/thesis/`
- §6 dir × archetype 分布 → `scripts/05_dir_archetype_distribution.py`
- §8 全 dir 網羅可視化 → `scripts/07_all_dirs_predictability.py`
- §9 dir メタ回帰 → `scripts/08_dir_meta_regression.py`
- §13 dir × 遷移 attrition → `scripts/12_horizon_dir_attrition.py`

詳細は `results/README.md` の §3, §5, §6, §8, §9, §13 を参照。

---

## 追補 (2026-05-30): ペア化の純粋な利得と per-dir 推薦価値

### B-1. dev-only / dir-only / pair の AUC 比較 [★主要追加]

「(dev, dir) ペアで予測する」設計の純粋な利得を初めて定量化する。`pair_trajectories.csv` の prob を group by 集約して 3 種を同サンプルで AUC 比較した。

| horizon | model | pair | dev-only | dir-only | **lift vs dev** | **lift vs dir** |
|---|---|---:|---:|---:|---:|---:|
| 0-3m | cold IRL | 0.779 | 0.712 | 0.691 | +0.067 | +0.087 |
| 0-3m | RF | 0.792 | 0.718 | 0.679 | +0.074 | +0.113 |
| 0-3m | アンサンブル | 0.813 | 0.727 | 0.697 | **+0.086** | **+0.117** |
| 3-6m | アンサンブル | 0.819 | 0.724 | 0.709 | **+0.095** | +0.110 |
| 6-9m | アンサンブル | 0.816 | 0.734 | 0.691 | +0.081 | **+0.124** |
| 9-12m | アンサンブル | 0.815 | 0.723 | 0.708 | +0.092 | +0.108 |

- ペア化で AUC が dev 平均に対し **+6.5~9.6pp**、dir 平均に対し **+6.7~12.4pp** 上昇。
- アンサンブルでペア化利得が最大化 (+8~10pp vs dev / +11~12pp vs dir)。
- bot 影響: |delta lift| max=0.0038, mean=0.0010 → 完全に頑健。

→ **論文 §5.1 / §5.2 で「(dev, dir) ペア設計」の存在意義を量的に裏付ける核心的根拠**。

出典: `scripts/16_pair_vs_baseline.py` / `results/baseline_comparison.csv` / `fig_baseline_comparison.pdf`

### B-3. per-dir Top-K Recall (推薦シナリオでの利点)

各 dir 内で dev を prob で sort → 上位 K に正例 dev が何人いるか。「次にこの dir をレビューしそうな人」の特定精度。

| horizon | model | Recall@10 | Recall@25 | Recall@50 |
|---|---|---:|---:|---:|
| 0-3m | アンサンブル | **0.876** | 0.967 | 0.991 |
| 3-6m | アンサンブル | 0.837 | 0.963 | 0.991 |
| 6-9m | アンサンブル | 0.859 | 0.966 | 0.992 |
| 9-12m | アンサンブル | **0.874** | 0.962 | 0.991 |

- per-dir Recall@10 で Ens (0.84~0.88) が IRL (0.84~0.87) をわずかに上回る。
- 「dir ごとに次に承諾しそうな dev top 10 を絞る」用途で Ens が最良。
- bot 影響: |delta HR@50| = 0 → 完全に頑健。

出典: `scripts/20_topk_recommendation.py` / `results/topk_per_dir_metrics.csv` / `fig_topk_recommendation.pdf`

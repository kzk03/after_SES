# SES 発表 補足資料

`scripts/analyze/study/supplement/` の各スクリプトで自動生成した図表の索引。
図表の数値はすべて実出力ファイルから読んでいる（出所列を参照）。

## 再生成

```
.venv/bin/python scripts/analyze/study/supplement/a_feature_catalog.py
.venv/bin/python scripts/analyze/study/supplement/b_archetype.py
.venv/bin/python scripts/analyze/study/supplement/c_period_auc.py
.venv/bin/python scripts/analyze/study/supplement/c_performance.py
.venv/bin/python scripts/analyze/study/supplement/d_setup.py
.venv/bin/python scripts/analyze/study/supplement/build_pptx.py
```

## 索引

スライドは A. 特徴量カタログ → B. 原型分析（ペア単位） → C. 原型分析（開発者単位） → D. 予測の深掘り → 付録 の順に並べている（順番は `build_pptx.py` の `CATEGORIES` で定義）。
旧ID は図表ファイル名（`fig/`・`tab/` の basename）。

| ID | 旧ID | 種別 | タイトル | ファイル | 出所 |
|---|---|---|---|---|---|
| S-A0 | — | 章扉 | A. 特徴量カタログ | — | — |
| S-A1 | S-A1 | 図 | 特徴量の全体像（37次元の内訳） | `outputs/analysis/ses_supplement/fig/S-A1.png` | src/review_predictor/IRL/features/common_features.py:113-143 ほか（次元数は実装から自動集計） |
| S-A2 | S-A2 | 表 | 開発者状態 26次元 ①（在籍・量／直近活動） | `outputs/analysis/ses_supplement/tab/S-A2.csv` | src/review_predictor/IRL/features/common_features.py:113-143 |
| S-A3 | S-A3 | 表 | 開発者状態 26次元 ②（承諾傾向／役割・ネットワーク） | `outputs/analysis/ses_supplement/tab/S-A3.csv` | src/review_predictor/IRL/features/common_features.py:113-143 |
| S-A4 | S-A4 | 表 | ディレクトリ親和度 6次元 ＋ 活動特徴 5次元 | `outputs/analysis/ses_supplement/tab/S-A4.csv` | src/review_predictor/IRL/features/path_features.py:81-88 / src/review_predictor/IRL/features/common_features.py:145-151 |
| S-A5 | S-A5 | 表 | 特徴量の正規化（すべて 0–1 に揃える） | `outputs/analysis/ses_supplement/tab/S-A5.csv` | src/review_predictor/IRL/features/common_features.py:176-207, 748-781 |
| S-B0 | — | 章扉 | B. 原型分析の深掘り（ペア単位） | — | — |
| S-B1 | S-B1 | 図 | 原型分析の定式化と k-means との違い | `outputs/analysis/ses_supplement/fig/S-B1.png` | scripts/analyze/study/archetype_irl_rf.py:75（左は概念図・実データではない） |
| S-B2 | S-B2 | 図 | エルボー法による原型数 K の決定（4系統） | `outputs/analysis/ses_supplement/fig/S-B2.png` | outputs/**/archetype/**/rss_elbow.csv |
| S-B2t | S-B2t | 表 | エルボーの実数値（RSS/N） | `outputs/analysis/ses_supplement/tab/S-B2t.csv` | outputs/**/archetype/**/rss_elbow.csv |
| S-B2n | S-B2n | 表 | 原型ごとのユニット数（K=4） | `outputs/analysis/ses_supplement/tab/S-B2n.csv` | outputs/**/k4_hard*metrics.csv（alpha CSV の dominant 件数と一致を確認） |
| S-B3 | S-B3 | 図 | 全特徴量シグネチャ（OpenStack・ペア単位） | `outputs/analysis/ses_supplement/fig/S-B3.png` | outputs/analysis/archetype_pair_level_volat_ep150_coldonly/k4_archetype_profile.csv |
| S-B4 | S-B4 | 図 | 全特徴量シグネチャ（AOSP・ペア単位） | `outputs/analysis/ses_supplement/fig/S-B4.png` | outputs/aosp32/archetype/pair/k4_archetype_profile.csv |
| S-B5 | S-B7 | 図 | 原型別の全特徴量プロファイル（OpenStack・ペア単位） | `outputs/analysis/ses_supplement/fig/S-B7.png` | outputs/analysis/archetype_pair_level_volat_ep150_coldonly/k4_archetype_profile.csv |
| S-B6 | S-B8 | 図 | 原型別の全特徴量プロファイル（AOSP・ペア単位） | `outputs/analysis/ses_supplement/fig/S-B8.png` | outputs/aosp32/archetype/pair/k4_archetype_profile.csv |
| S-B7 | S-B11 | 図 | 所属度 α の分布 | `outputs/analysis/ses_supplement/fig/S-B11.png` | outputs/**/k4_*_alpha_with_metrics.csv |
| S-B7t | S-B11t | 表 | 所属度 α の要約 | `outputs/analysis/ses_supplement/tab/S-B11t.csv` | outputs/**/k4_*_alpha_with_metrics.csv |
| S-B8 | S-B12 | 図 | K を変えたときの原型サイズ構成比 | `outputs/analysis/ses_supplement/fig/S-B12.png` | outputs/**/k{2..6}_hard*metrics.csv |
| S-B9 | S-B13 | 図 | K を変えても原型間の得意不得意は消えない | `outputs/analysis/ses_supplement/fig/S-B13.png` | outputs/**/k{2..6}_pooled_auc*.csv |
| S-B9t | S-B13t | 表 | K別・原型別の Δ(IRL−RF) | `outputs/analysis/ses_supplement/tab/S-B13t.csv` | outputs/**/k{2..6}_pooled_auc*.csv |
| S-B10 | S-B14 | 図 | 2プロジェクトの原型はどこまで同じか | `outputs/analysis/ses_supplement/fig/S-B14.png` | outputs/**/k4_archetype_profile.csv |
| S-B10t | S-B14t | 表 | 原型対応の cosine 類似度 | `outputs/analysis/ses_supplement/tab/S-B14t.csv` | outputs/**/k4_archetype_profile.csv |
| S-C0 | — | 章扉 | C. 原型分析の深掘り（開発者単位） | — | — |
| S-C1 | S-B5 | 図 | 全特徴量シグネチャ（OpenStack・開発者単位） | `outputs/analysis/ses_supplement/fig/S-B5.png` | outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_archetype_profile.csv |
| S-C2 | S-B6 | 図 | 全特徴量シグネチャ（AOSP・開発者単位） | `outputs/analysis/ses_supplement/fig/S-B6.png` | outputs/aosp32/archetype/dev/k4_archetype_profile.csv |
| S-C3 | S-B9 | 図 | 原型別の全特徴量プロファイル（OpenStack・開発者単位） | `outputs/analysis/ses_supplement/fig/S-B9.png` | outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_archetype_profile.csv |
| S-C4 | S-B10 | 図 | 原型別の全特徴量プロファイル（AOSP・開発者単位） | `outputs/analysis/ses_supplement/fig/S-B10.png` | outputs/aosp32/archetype/dev/k4_archetype_profile.csv |
| S-C5 | S-B15 | 図 | 開発者単位とペア単位の原型の関係 | `outputs/analysis/ses_supplement/fig/S-B15.png` | outputs/**/k4_dev_alpha_with_metrics.csv × k4_pair_alpha_with_metrics.csv |
| S-C5t | S-B15t | 表 | 開発者単位×ペア単位のクロス集計 | `outputs/analysis/ses_supplement/tab/S-B15t.csv` | outputs/**/k4_*_alpha_with_metrics.csv |
| S-D0 | — | 章扉 | D. 予測の深掘り | — | — |
| S-D1 | S-C10 | 図 | IRL（MCE-IRL）の定式化 | `outputs/analysis/ses_supplement/fig/S-C10.png` | src/review_predictor/IRL/model/mce_irl_predictor.py:10-26,:83-87,:1039-1068 / scripts/train/train_mce_irl.py:630-637 |
| S-D2 | S-C1 | 図 | 原型×期間の Δ(IRL−RF)（ペア単位） | `outputs/analysis/ses_supplement/fig/S-C1.png` | outputs/analysis/ses_supplement/tab/period_auc_openstack_pair.csv（pooled_pairs_blabel_t0_20230101.csv から今回算出） ／ outputs/aosp32/archetype/pair/k4_period_auc.csv |
| S-D2t | S-C1t | 表 | 原型×期間の AUC 実数値（ペア単位） | `outputs/analysis/ses_supplement/tab/S-C1t.csv` | outputs/analysis/ses_supplement/tab/period_auc_openstack_pair.csv（pooled_pairs_blabel_t0_20230101.csv から今回算出） ／ outputs/aosp32/archetype/pair/k4_period_auc.csv |
| S-D3 | S-C2 | 図 | 原型×期間の Δ(IRL−RF)（開発者単位） | `outputs/analysis/ses_supplement/fig/S-C2.png` | outputs/analysis/ses_supplement/tab/period_auc_openstack_dev.csv（pooled_pairs_blabel_t0_20230101.csv から今回算出） ／ outputs/aosp32/archetype/dev/k4_period_auc.csv |
| S-D3t | S-C2t | 表 | 原型×期間の AUC 実数値（開発者単位） | `outputs/analysis/ses_supplement/tab/S-C2t.csv` | outputs/analysis/ses_supplement/tab/period_auc_openstack_dev.csv（pooled_pairs_blabel_t0_20230101.csv から今回算出） ／ outputs/aosp32/archetype/dev/k4_period_auc.csv |
| S-D4 | S-C3 | 図 | アンサンブルはどこで効いたか（OpenStack） | `outputs/analysis/ses_supplement/fig/S-C3.png` | outputs/analysis/ses_supplement/tab/period_auc_openstack_{pair,dev}.csv（AOSP はラベル定義が既存CSVと異なるため未算出） |
| S-D5 | S-C4 | 図 | 原型別 pooled AUC と 95%CI（ペア単位） | `outputs/analysis/ses_supplement/fig/S-C4.png` | outputs/analysis/archetype_pair_level_volat_ep150_coldonly/k4_pooled_auc.csv ／ outputs/aosp32/archetype/pair/k4_pooled_auc.csv（OpenStack 側の warm_irl は除外） |
| S-D5t | S-C4t | 表 | 原型別 pooled AUC の実数値（ペア単位） | `outputs/analysis/ses_supplement/tab/S-C4t.csv` | outputs/**/k4_pooled_auc*.csv |
| S-D6 | S-C5 | 図 | 原型別 pooled AUC と 95%CI（開発者単位） | `outputs/analysis/ses_supplement/fig/S-C5.png` | outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_pooled_auc_bootstrap.csv ／ outputs/aosp32/archetype/dev/k4_pooled_auc_bootstrap.csv（OpenStack 側の warm_irl は除外） |
| S-D6t | S-C5t | 表 | 原型別 pooled AUC の実数値（開発者単位） | `outputs/analysis/ses_supplement/tab/S-C5t.csv` | outputs/**/k4_pooled_auc*.csv |
| S-D7 | S-C6 | 表 | 原型別 Wilcoxon 符号順位検定（IRL vs RF・ペア単位） | `outputs/analysis/ses_supplement/tab/S-C6.csv` | outputs/analysis/archetype_pair_level_volat_ep150_coldonly/k4_wilcoxon.csv ／ outputs/aosp32/archetype/pair/k4_wilcoxon.csv（OpenStack の warm_irl 比較・acc 比較の行は除外） |
| S-D8 | S-C7 | 表 | 原型別 Wilcoxon 符号順位検定（IRL vs RF・開発者単位） | `outputs/analysis/ses_supplement/tab/S-C7.csv` | outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_wilcoxon.csv ／ outputs/aosp32/archetype/dev/k4_wilcoxon.csv（OpenStack の warm_irl 比較・acc 比較の行は除外） |
| S-D9 | S-C8 | 図 | アブレーション：LSTM と IRL の比較 | `outputs/analysis/ses_supplement/fig/S-C8.png` | outputs/ses_review/b3_ablation/headline_blabel_*.csv ／ docs/SES発表用_アブレーション.xlsx（AOSP） |
| S-D9t | S-C8t | 表 | アブレーションの実数値 | `outputs/analysis/ses_supplement/tab/S-C8t.csv` | outputs/ses_review/b3_ablation/headline_blabel_*.csv ／ docs/SES発表用_アブレーション.xlsx |
| S-D10 | S-C9 | 図 | アンサンブルの期間別内訳 | `outputs/analysis/ses_supplement/fig/S-C9.png` | outputs/analysis/ses_supplement/tab/overall_auc_{openstack,aosp}.csv（OpenStack は headline CSV と一致を確認済み） |
| S-D10t | S-C9t | 表 | アンサンブルの実数値 | `outputs/analysis/ses_supplement/tab/S-C9t.csv` | outputs/analysis/ses_supplement/tab/overall_auc_*.csv |
| S-E0 | — | 章扉 | 付録. 実験設定・データセット | — | — |
| S-E1 | S-D1 | 表 | データセットの規模 | `outputs/analysis/ses_supplement/tab/S-D1.csv` | data/combined_raw_main32.csv ／ SES/ses_main.tex 表1 ／ outputs/**/train_0-3m/eval_0-3m/pair_predictions.csv |
| S-E2 | S-D2 | 表 | ラベル定義（依頼の有無で負例を2種類に分ける） | `outputs/analysis/ses_supplement/tab/S-D2.csv` | SES/ses_main.tex:318（弱負例重みの適用範囲）／ 発表スライド10枚目 |
| S-E3 | S-D3 | 表 | 評価設計 | `outputs/analysis/ses_supplement/tab/S-D3.csv` | scripts/variant/run_ses_b3_ablation_queue.sh:15-37 ／ SES/ses_main.tex |
| S-E4 | S-D4 | 表 | モデルの学習設定 | `outputs/analysis/ses_supplement/tab/S-D4.csv` | 各スクリプトの該当行（右列） |
| S-E5 | S-D5 | 表 | 原型分析の対象（t0直前6か月に履歴があるユニット） | `outputs/analysis/ses_supplement/tab/S-D5.csv` | outputs/**/k4_{dev,pair}_alpha_with_metrics.csv の行数 |

スライド: `docs/SES補足資料_自動生成.pptx`（60 枚）

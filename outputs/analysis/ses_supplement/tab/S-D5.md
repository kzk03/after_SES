# 原型分析の対象（t0直前6か月に履歴があるユニット）

⚠ 発表スライド18枚目は OpenStack のペアを 3,747 としているが、実際に原型を当てているペアは 3,677（dev 側の 3,747 と取り違えている可能性）。AOSP は 21,291 で一致。

| プロジェクト    | 粒度    |   原型分析の対象数 |   特徴量の次元 | 元ファイル                                                                                     |
|:----------|:------|-----------:|---------:|:------------------------------------------------------------------------------------------|
| OpenStack | 開発者単位 |        289 |       26 | outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_dev_alpha_with_metrics.csv      |
| OpenStack | ペア単位  |      3,677 |       32 | outputs/analysis/archetype_pair_level_volat_ep150_coldonly/k4_pair_alpha_with_metrics.csv |
| AOSP      | 開発者単位 |      1,248 |       26 | outputs/aosp32/archetype/dev/k4_dev_alpha_with_metrics.csv                                |
| AOSP      | ペア単位  |     21,291 |       32 | outputs/aosp32/archetype/pair/k4_pair_alpha_with_metrics.csv                              |

出所: outputs/**/k4_{dev,pair}_alpha_with_metrics.csv の行数

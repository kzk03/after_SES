# データセットの規模

AOSP のデータセット全体の統計は、生データ(combined_raw_aosp32.csv)が手元に無いため論文の表から転記している。

| 項目            |   OpenStack |    AOSP | 確認状況                                                  |
|:--------------|------------:|--------:|:------------------------------------------------------|
| リポジトリ数        |          32 |      32 | OpenStack は data/combined_raw_main32.csv から再計算し論文値と一致 |
| チケット（change）数 |      42,441 | 408,574 | OpenStack は data/combined_raw_main32.csv から再計算し論文値と一致 |
| レビュー依頼行数      |     161,111 | 574,810 | OpenStack は data/combined_raw_main32.csv から再計算し論文値と一致 |
| レビュアー数        |       2,372 |   8,901 | OpenStack は data/combined_raw_main32.csv から再計算し論文値と一致 |
| 予測対象の開発者数     |         436 |   2,508 | 両プロジェクトとも pair_predictions.csv から再計算                  |
| 予測対象のペア数      |       4,534 |  26,830 | 両プロジェクトとも pair_predictions.csv から再計算                  |

出所: data/combined_raw_main32.csv ／ SES/ses_main.tex 表1 ／ outputs/**/train_0-3m/eval_0-3m/pair_predictions.csv

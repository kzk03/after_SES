# -*- coding: utf-8 -*-
"""補足資料 D群: 実験設定・データセット.

ローカルで再現できた数値と、論文/発表資料からの転記を必ず区別して出す。

実行: .venv/bin/python scripts/analyze/study/supplement/d_setup.py
"""
from __future__ import annotations

import pathlib
import re
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import ROOT, SOURCES, save_section, save_table, write_manifest  # noqa: E402

RAW_OS = ROOT / "data" / "combined_raw_main32.csv"
TEX = ROOT / "SES" / "ses_main.tex"
PRED = {
    "openstack": ROOT / "outputs/dynstartB_ep150/lstm_baseline/train_0-3m/eval_0-3m/pair_predictions.csv",
    "aosp": ROOT / "outputs/aosp32/dynstartB_ep150/lstm_baseline/train_0-3m/eval_0-3m/pair_predictions.csv",
}


def tex_dataset_table() -> dict[str, tuple[str, str]]:
    """論文 表1 (tab:dataset) の行を読む。{項目: (OpenStack, AOSP)}。"""
    text = TEX.read_text(encoding="utf-8")
    out = {}
    for item in ["リポジトリ数", "チケット（change）数", "レビュー依頼行数", "レビュアー数"]:
        m = re.search(re.escape(item) + r"\s*&\s*([0-9{},]+)\s*&\s*([0-9{},]+)\s*\\\\", text)
        assert m, f"論文の表から {item} を読めない"
        out[item] = (m.group(1).replace("{,}", ","), m.group(2).replace("{,}", ","))
    return out


def table_d1_dataset() -> None:
    tex = tex_dataset_table()
    raw = pd.read_csv(RAW_OS, usecols=["change_id", "project", "email"])
    os_local = {
        "リポジトリ数": f"{raw['project'].nunique():,}",
        "チケット（change）数": f"{raw['change_id'].nunique():,}",
        "レビュー依頼行数": f"{len(raw):,}",
        "レビュアー数": f"{raw['email'].nunique():,}",
    }
    rows = []
    for item, (os_tex, ao_tex) in tex.items():
        local = os_local[item]
        rows.append({
            "項目": item,
            "OpenStack": local,
            "AOSP": ao_tex,
            "確認状況": ("OpenStack は data/combined_raw_main32.csv から再計算し論文値と一致"
                     if local == os_tex else
                     f"⚠ OpenStack 再計算={local} が論文値={os_tex} と不一致"),
        })
    # 予測対象 (train/eval 両期間で観測されたペア)
    for label, key in [("予測対象の開発者数", "developer"), ("予測対象のペア数", None)]:
        vals = {}
        for ds, p in PRED.items():
            d = pd.read_csv(p, usecols=["developer", "directory"])
            vals[ds] = (f"{d['developer'].nunique():,}" if key
                        else f"{len(d):,}")
        rows.append({"項目": label, "OpenStack": vals["openstack"],
                     "AOSP": vals["aosp"],
                     "確認状況": "両プロジェクトとも pair_predictions.csv から再計算"})
    save_table(pd.DataFrame(rows), "S-D1", "データセットの規模",
               "AOSP のデータセット全体の統計は、生データ(combined_raw_aosp32.csv)が"
               "手元に無いため論文の表から転記している。",
               f"{RAW_OS.relative_to(ROOT)} ／ SES/ses_main.tex 表1 ／ "
               "outputs/**/train_0-3m/eval_0-3m/pair_predictions.csv",
               col_widths=[3.0, 2.0, 2.0, 6.0], font_size=11)


def table_d2_label() -> None:
    rows = [
        {"ラベル": "+1（正例）", "条件": "その期間にレビュー依頼があり、承諾（貢献）した",
         "重み": "1.0", "備考": "予測したい事象"},
        {"ラベル": "0（強い負例）", "条件": "レビュー依頼はあったが貢献しなかった",
         "重み": "1.0", "備考": "機会があったのに選ばなかった"},
        {"ラベル": "0*（弱い負例）", "条件": "そもそもレビュー依頼が来ていない",
         "重み": "0.5", "備考": "IRL の損失にのみ適用。RF の学習には適用しない"},
    ]
    save_table(pd.DataFrame(rows), "S-D2", "ラベル定義（依頼の有無で負例を2種類に分ける）",
               "「依頼が来ていないだけ」と「依頼を断った」を同じ負例にしないための重み付け。",
               "SES/ses_main.tex:318（弱負例重みの適用範囲）／ 発表スライド10枚目",
               col_widths=[2.4, 5.6, 1.2, 4.4], font_size=12)

    rows2 = [
        {"項目": "予測時点 t0", "設定": "2023-01-01"},
        {"項目": "訓練期間", "設定": "3年（実装では horizon ごとに学習窓を33ステップへ等化）"},
        {"項目": "評価期間", "設定": "1年を3か月ずつ4分割（W1:1–3m / W2:4–6m / W3:7–9m / W4:10–12m）"},
        {"項目": "リーク防止", "設定": "train_end = t0 − 12か月。ラベル判定のみ後方に許す"},
        {"項目": "評価指標", "設定": "AUC-ROC（bootstrap による95%信頼区間つき）"},
        {"項目": "予測単位", "設定": "開発者 × ディレクトリのペア"},
    ]
    save_table(pd.DataFrame(rows2), "S-D3", "評価設計",
               "「訓練期間3年」は本文の記述。実装では 3–6m 以降の horizon で "
               "train_start を前倒しし、学習ステップ数を 33 に揃えている"
               "（scripts/variant/run_ses_b3_ablation_queue.sh:17-23 に明記）。",
               "scripts/variant/run_ses_b3_ablation_queue.sh:15-37 ／ SES/ses_main.tex",
               col_widths=[3.0, 9.0], font_size=12)


def table_d4_models() -> None:
    rows = [
        {"モデル": "IRL（提案）", "設定": "MCE IRL / epochs=150 / 学習率 3e-4 / 乱数シード 777 / "
                                   "状態32次元（開発者状態26＋ディレクトリ親和度6）",
         "出所": "scripts/variant/run_mce_irl_variant_single.sh:77, "
               "scripts/train/train_mce_irl.py:1544, :1495"},
        {"モデル": "RF（比較）", "設定": "決定木100本 / 深さ無制限 / 葉あたり最小サンプル1 / "
                                  "class_weight=balanced / seed 777 / 探索なし / 入力37次元",
         "出所": "scripts/analyze/eval/eval_mce_irl_path_prediction.py:1157, "
               "SES/ses_main.tex:318"},
        {"モデル": "LSTM（アブレーション）", "設定": "FINAL_STEP_ONLY=1 の退化版"
                                          "（最終有効ステップのみ教師）/ epochs=150 / IRL と同条件",
         "出所": "scripts/variant/run_ses_b3_ablation_queue.sh:5-8, :36"},
        {"モデル": "アンサンブル", "設定": "IRL と RF の出力確率の単純平均 (p_IRL + p_RF)/2",
         "出所": "scripts/analyze/study/verify_blabel_headline.py:87"},
        {"モデル": "（未確認）", "設定": "IRL の隠れ層次元・バッチサイズはスクリプトから特定できず未確認",
         "出所": "—"},
    ]
    save_table(pd.DataFrame(rows), "S-D4", "モデルの学習設定",
               "確認できなかった項目は「未確認」と明記している（推測で埋めていない）。",
               "各スクリプトの該当行（右列）",
               col_widths=[2.6, 6.4, 4.2], font_size=10)


def table_d5_archetype_target() -> None:
    rows = []
    for (ds, lv), src in SOURCES.items():
        alpha = pd.read_csv(src.path(src.alpha_file),
                            index_col=0 if lv == "dev" else None)
        rows.append({
            "プロジェクト": "OpenStack" if ds == "openstack" else "AOSP",
            "粒度": "開発者単位" if lv == "dev" else "ペア単位",
            "原型分析の対象数": f"{len(alpha):,}",
            "特徴量の次元": 26 if lv == "dev" else 32,
            "元ファイル": src.rel(src.alpha_file),
        })
    save_table(pd.DataFrame(rows), "S-D5", "原型分析の対象（t0直前6か月に履歴があるユニット）",
               "⚠ 発表スライド18枚目は OpenStack のペアを 3,747 としているが、"
               "実際に原型を当てているペアは 3,677（dev 側の 3,747 と取り違えている可能性）。"
               "AOSP は 21,291 で一致。",
               "outputs/**/k4_{dev,pair}_alpha_with_metrics.csv の行数",
               col_widths=[2.4, 2.2, 2.6, 2.0, 6.0], font_size=11)


def main() -> None:
    save_section("S-D0", "D. 実験設定・データセット",
                 "データ規模・ラベル定義・評価設計・モデル設定・原型分析の対象")
    table_d1_dataset()
    table_d2_label()
    table_d4_models()
    table_d5_archetype_target()
    write_manifest("d")


if __name__ == "__main__":
    main()

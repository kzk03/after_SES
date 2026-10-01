# -*- coding: utf-8 -*-
"""補足資料 A群: 特徴量カタログ (37次元の全リスト・定義・正規化).

特徴量名と正規化の設定はすべて src/review_predictor/IRL/features/ から
import して生成する (手で並べない)。説明文 FEATURE_DOC のキーが実装の
FEATURE_NAMES_WITH_PATH と一致することを起動時に assert する。

実行: .venv/bin/python scripts/analyze/study/supplement/a_feature_catalog.py
"""
from __future__ import annotations

import sys

import matplotlib.patches as mpatches
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    FEATURE_GROUPS, GROUP_OF, ROOT, new_fig, save_fig, save_section, save_table,
    write_manifest,
)

sys.path.insert(0, str(ROOT / "src"))
from review_predictor.IRL.features.common_features import (  # noqa: E402
    _LOG1P_FEATURES, _NORM_CAPS, ACTION_FEATURES, FEATURE_NAMES_WITH_PATH,
    STATE_FEATURES,
)
from review_predictor.IRL.features.path_features import (  # noqa: E402
    PATH_FEATURE_NAMES,
)

SKIP_NORM = {"activity_trend", "response_time_trend", "accept_rate_drift_30d"}

SRC_STATE = "src/review_predictor/IRL/features/common_features.py:113-143"
SRC_PATH = "src/review_predictor/IRL/features/path_features.py:81-88"
SRC_ACTION = "src/review_predictor/IRL/features/common_features.py:145-151"
SRC_NORM = "src/review_predictor/IRL/features/common_features.py:176-207, 748-781"

# 各特徴量の 1 行定義 (実装の計算内容を要約)
FEATURE_DOC: dict[str, str] = {
    # ── 開発者状態 26 ──
    "window_tenure_days": "特徴量窓の中での初回活動日から窓終端までの日数",
    "total_changes": "窓内に自分が投稿した変更 (owner) の件数",
    "total_reviews": "窓内に自分が受けたレビュー依頼 (reviewer) の件数",
    "recent_activity_frequency": "直近30日の活動件数 ÷ 30（1日あたり件数）",
    "avg_activity_gap": "活動と活動の間隔の中央値（日）",
    "activity_trend": "窓を時間で前半／後半に二分したときの活動量の増減",
    "unique_collaborator_count": "窓内に関わったユニークな owner 数のスコア",
    "overall_acceptance_rate": "窓内の全依頼に対する承諾（label=1）の割合",
    "recent_acceptance_rate": "直近30日の依頼に対する承諾の割合",
    "recent_load_ratio_30d_all": "直近30日の件数 ÷ 窓全体の30日あたり平均件数",
    "days_since_last_activity": "最終活動からの経過日数",
    "reciprocity_score": "自分がレビューした owner のうち、自分の変更もレビューした人の割合",
    "recent_load_ratio_7d_30d": "直近7日の1日あたり件数 ÷ 直近30日の1日あたり件数",
    "core_reviewer_ratio": "自分のレビュー件数 ÷ プロジェクト全体の依頼件数",
    "acceptance_rate_last10": "直近10件の依頼における承諾の割合",
    "active_months_ratio": "活動があった月数 ÷ 窓の月数",
    "response_time_trend": "直近30日の応答速度 − 窓全体の応答速度（速度=1/(1+日数/3)）",
    "reviewer_owner_ratio": "レビュー件数 ÷（レビュー件数＋投稿件数）＝役割バランス",
    "n_projects": "窓内に活動したユニークな project 数",
    "cross_project_review_share": "ホームプロジェクト以外でのレビュー比率",
    "same_domain_share": "owner と email ドメインが一致する依頼の比率",
    "recent_switch_rate": "直近10件での承諾⇄拒否の切り替わり率（変動の大きさ）",
    "time_since_last_accept": "最後に承諾（label=1）してからの経過日数",
    "time_since_last_reject": "最後に拒否（label=0）してからの経過日数",
    "accept_rate_drift_30d": "直近30日の承諾率 − 窓全体の承諾率",
    "concurrent_open_reviews_7d": "直近7日に受けた依頼のうち未応答の件数（瞬間負荷）",
    # ── ディレクトリ親和度 6 ──
    "path_review_count": "対象ディレクトリと交差する過去のレビュー件数",
    "path_recency": "対象ディレクトリを最後に触ってからの経過日数（近いほど1に近い）",
    "path_acceptance_rate": "対象ディレクトリでの過去の承諾率",
    "path_owner_overlap": "対象ディレクトリの過去 owner 集合と自分の協働相手の Jaccard 係数",
    "path_hub_score": "対象ディレクトリの co-change ネットワークにおける hub スコアの最大値",
    "path_neighbor_coverage": "co-change 近傍ディレクトリのうち自分が担当している割合",
    # ── 活動特徴 5 ──
    "avg_action_intensity": "1件あたりの変更ファイル数と変更行数の幾何平均",
    "avg_change_lines": "1件あたりの変更行数（追加＋削除）の平均",
    "avg_response_time": "平均応答日数を速度に変換した値（1/(1+日数/3)）",
    "avg_review_size": "1件あたりの平均ファイル数（10ファイルで1.0）",
    "repeat_collaboration_rate": "複数回依頼してきた owner の割合（協働の反復率）",
}

CATEGORIES = [
    ("開発者状態", STATE_FEATURES, SRC_STATE, "#E64B35"),
    ("ディレクトリ親和度", PATH_FEATURE_NAMES, SRC_PATH, "#4DBBD5"),
    ("活動特徴", ACTION_FEATURES, SRC_ACTION, "#00A087"),
]
CAT_OF = {f: c for c, feats, _, _ in CATEGORIES for f in feats}
CAT_COLOR = {c: col for c, _, _, col in CATEGORIES}
SHORT_CAT = {"開発者状態": "状態", "ディレクトリ親和度": "親和度", "活動特徴": "活動"}


def norm_desc(name: str) -> str:
    """実装 (normalize_features) から正規化方式の説明を生成する。"""
    if name in PATH_FEATURE_NAMES:
        if name == "path_review_count":
            return "log1p / log1p(100)（path_features 内で正規化）"
        if name == "path_hub_score":
            return "64 で除算（path_features 内で正規化）"
        return "0–1（定義上そのまま）"
    if name in SKIP_NORM:
        return "[-1, +1]（正規化の対象外）"
    if name in _LOG1P_FEATURES:
        return f"log1p / log1p({_LOG1P_FEATURES[name]:g})"
    if name in _NORM_CAPS:
        return f"{_NORM_CAPS[name]:g} で除算しクリップ"
    return "0–1（計算時点で 0–1）"


def check_consistency() -> None:
    impl = list(FEATURE_NAMES_WITH_PATH)
    assert len(impl) == 37, f"実装の特徴量が37次元ではない: {len(impl)}"
    missing = [f for f in impl if f not in FEATURE_DOC]
    extra = [f for f in FEATURE_DOC if f not in impl]
    assert not missing and not extra, f"FEATURE_DOC が実装とずれている: {missing=} {extra=}"
    print(f"[check] FEATURE_DOC と実装の {len(impl)} 特徴量が一致")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig_a1_overview() -> None:
    """S-A1: 37次元の内訳と、どのモデルがどの部分を使うか。"""
    uses = [
        ("RF（比較手法）の入力", ["開発者状態", "ディレクトリ親和度", "活動特徴"]),
        ("IRL（提案手法）の状態 $s_t$", ["開発者状態", "ディレクトリ親和度"]),
        ("原型分析・ペア単位", ["開発者状態", "ディレクトリ親和度"]),
        ("原型分析・開発者単位", ["開発者状態"]),
    ]
    dims = {c: len(feats) for c, feats, _, _ in CATEGORIES}

    fig = new_fig(12.8, 6.2)
    ax = fig.add_axes([0.24, 0.13, 0.70, 0.72])
    ax.set_xlim(0, 37)
    ax.set_ylim(-0.6, len(uses) - 0.4)
    ax.set_yticks(range(len(uses)))
    ax.set_yticklabels([u[0] for u in uses])
    ax.set_xlabel("特徴量の次元数")
    ax.set_xticks([0, 10, 20, 26, 32, 37])
    ax.set_xticklabels(["0", "10", "20", "26", "32", "37"])
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    for y, (_, cats) in enumerate(uses):
        left = 0.0
        for c in ["開発者状態", "ディレクトリ親和度", "活動特徴"]:
            if c not in cats:
                continue
            w = dims[c]
            ax.barh(y, w - 0.12, left=left, height=0.52, color=CAT_COLOR[c],
                    edgecolor="white", linewidth=0, zorder=3)
            # 幅の狭いセグメントは短縮名にして、はみ出しを防ぐ
            inner = f"{c}\n{w}次元" if w >= 10 else f"{SHORT_CAT[c]}\n{w}次元"
            ax.text(left + w / 2, y, inner, ha="center", va="center",
                    color="white", fontsize=12 if w >= 10 else 11,
                    fontweight="bold", zorder=4)
            left += w
        ax.text(left + 0.35, y, f"計 {int(left)} 次元", va="center", fontsize=14,
                fontweight="bold", color="#333333")

    handles = [mpatches.Patch(color=CAT_COLOR[c], label=f"{c}（{dims[c]}次元）")
               for c in ["開発者状態", "ディレクトリ親和度", "活動特徴"]]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.34),
              ncol=3, frameon=False)
    ax.set_title("3カテゴリ計37次元と、モデルごとに使う範囲", pad=14)
    fig.text(0.02, 0.945,
             "原型分析は活動特徴5次元を使わない（開発者単位26 / ペア単位32）",
             fontsize=13, color="#555555")
    save_fig(fig, "S-A1", "特徴量の全体像（37次元の内訳）",
             "RFは37次元すべて、IRLの状態は32次元、原型分析は活動特徴を除いた26/32次元。",
             f"{SRC_STATE} ほか（次元数は実装から自動集計）")


def _feature_table(feats: list[str]) -> pd.DataFrame:
    rows = []
    for f in feats:
        rows.append({
            "カテゴリ": CAT_OF[f],
            "グループ": GROUP_OF.get(f, "活動特徴"),
            "特徴量": f,
            "定義": FEATURE_DOC[f],
            "正規化": norm_desc(f),
        })
    return pd.DataFrame(rows)


def tables_a2_a4() -> None:
    """S-A2〜S-A4: 37特徴量の一覧表 (グループごとに分割)。"""
    g = {name: feats for name, feats in FEATURE_GROUPS}
    part1 = g["在籍・量"] + g["直近活動"]
    part2 = g["承諾傾向"] + g["役割・ネットワーク"]
    assert sorted(part1 + part2) == sorted(STATE_FEATURES), "開発者状態26の分割が不完全"

    save_table(_feature_table(part1)[["グループ", "特徴量", "定義", "正規化"]],
               "S-A2", "開発者状態 26次元 ①（在籍・量／直近活動）",
               "レビュー履歴の長さ・量と、直近の活動リズムを表す12次元。",
               SRC_STATE, col_widths=[1.5, 3.0, 5.6, 2.7], font_size=10)

    save_table(_feature_table(part2)[["グループ", "特徴量", "定義", "正規化"]],
               "S-A3", "開発者状態 26次元 ②（承諾傾向／役割・ネットワーク）",
               "承諾するかどうかの傾向と、プロジェクト内での役割・つながりを表す14次元。",
               SRC_STATE, col_widths=[1.8, 3.0, 5.3, 2.7], font_size=10)

    save_table(_feature_table(list(PATH_FEATURE_NAMES) + list(ACTION_FEATURES))
               [["カテゴリ", "特徴量", "定義", "正規化"]],
               "S-A4", "ディレクトリ親和度 6次元 ＋ 活動特徴 5次元",
               "ディレクトリ親和度がモジュール単位予測の中核。活動特徴5次元はRFのみが使う。",
               f"{SRC_PATH} / {SRC_ACTION}", col_widths=[2.0, 3.0, 5.3, 2.5],
               font_size=10)


def table_a5_normalization() -> None:
    """S-A5: 正規化方式の内訳。"""
    rows = []
    log1p_feats = [f for f in FEATURE_NAMES_WITH_PATH if f in _LOG1P_FEATURES]
    cap_feats = [f for f in FEATURE_NAMES_WITH_PATH
                 if f not in _LOG1P_FEATURES and f not in SKIP_NORM
                 and f not in PATH_FEATURE_NAMES and f in _NORM_CAPS]
    plain = [f for f in FEATURE_NAMES_WITH_PATH
             if f not in _LOG1P_FEATURES and f not in SKIP_NORM
             and f not in PATH_FEATURE_NAMES and f not in _NORM_CAPS]
    rows.append({
        "方式": "log1p + キャップ",
        "式": "log1p(x) / log1p(cap)",
        "対象": ", ".join(f"{f}({_LOG1P_FEATURES[f]:g})" for f in log1p_feats),
        "件数": len(log1p_feats),
    })
    rows.append({
        "方式": "キャップ除算",
        "式": "min(x / cap, 1)",
        "対象": ", ".join(f"{f}({_NORM_CAPS[f]:g})" for f in cap_feats),
        "件数": len(cap_feats),
    })
    rows.append({
        "方式": "正規化なし（[-1,+1]）",
        "式": "そのまま",
        "対象": ", ".join(sorted(SKIP_NORM)),
        "件数": len(SKIP_NORM),
    })
    rows.append({
        "方式": "計算時点で 0–1",
        "式": "—",
        "対象": ", ".join(plain),
        "件数": len(plain),
    })
    rows.append({
        "方式": "path 側で正規化",
        "式": "log1p(100) / 64 除算 / 定義上0–1",
        "対象": ", ".join(PATH_FEATURE_NAMES),
        "件数": len(PATH_FEATURE_NAMES),
    })
    df = pd.DataFrame(rows)
    total = int(df["件数"].sum())
    assert total == 37, f"正規化方式の合計が37にならない: {total}"
    save_table(df, "S-A5", "特徴量の正規化（すべて 0–1 に揃える）",
               "右裾が重いカウント系は log1p、比率系はキャップ除算。"
               "トレンド系3つだけ [-1,+1] のまま扱う。",
               SRC_NORM, col_widths=[2.4, 2.6, 6.6, 1.0], font_size=10)


def main() -> None:
    check_consistency()
    save_section("S-A0", "A. 特徴量カタログ",
                 "開発者の状態 $s_t$ を表す37次元の内訳・定義・正規化")
    fig_a1_overview()
    tables_a2_a4()
    table_a5_normalization()
    write_manifest("a")


if __name__ == "__main__":
    main()

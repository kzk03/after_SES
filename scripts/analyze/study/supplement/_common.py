# -*- coding: utf-8 -*-
"""SES 発表 補足資料の共通モジュール.

配色・原型ラベル・特徴量の並び順は既存の発表用スクリプトからの移植
(archetype_profile_eugster.py / archetype_profile_eugster_aosp.py /
archetype_pair_present.py)。既存スクリプトは変更しないため、値を変える
ときは両方を直すこと。

図に出す数値は必ず実出力 CSV から読む。ハードコードしない。
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import japanize_matplotlib  # noqa: F401
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs" / "analysis" / "ses_supplement"
FIG = OUT / "fig"
TAB = OUT / "tab"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 原型ラベル・配色 (発表スライドと同一)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# dev 単位の原型名 (archetype_profile_eugster.py:24-29 / _aosp.py:25-30)
DEV_NAMES = {
    "openstack": {
        "A1": "低活動型",
        "A2": "短期・レビュー中心型",
        "A3": "高承諾・高活動型",
        "A4": "常時活動・低承諾型",
    },
    "aosp": {
        "A1": "常時活動・高承諾型",
        "A2": "低活動型",
        "A3": "高承諾・レビュアー専業型",
        "A4": "低承諾型",
    },
}
# pair 単位の原型名 (archetype_pair_present.py:25-38)
PAIR_NAMES = {
    "openstack": {
        "P1": "他プロジェクト経験者",
        "P2": "関与低下型",
        "P3": "高承諾型",
        "P4": "低活動・低承諾型",
    },
    "aosp": {
        "P1": "高活動・未経験ディレクトリ型",
        "P2": "ディレクトリ経験者・高関与型",
        "P3": "低承諾型",
        "P4": "高承諾・汎用型",
    },
}
# OpenStack と AOSP の profile を cosine 類似度で対応づけた結果、
# P1<->P1 / P2<->P2 は番号一致、P3<->OS-P4 / P4<->OS-P3 はねじれて対応する。
# OpenStack 側は既存資料があるため変更せず、AOSP 側の P3/P4 の色を入れ替える。
# (archetype_pair_present.py:41-58 / archetype_profile_eugster_aosp.py:37-44)
PAIR_COLORS = {
    "openstack": {"P1": "#E64B35", "P2": "#4DBBD5", "P3": "#00A087", "P4": "#F39B7F"},
    "aosp": {"P1": "#E64B35", "P2": "#4DBBD5", "P3": "#F39B7F", "P4": "#00A087"},
}
DEV_COLORS = {
    ds: {"A1": "#E64B35", "A2": "#4DBBD5", "A3": "#00A087", "A4": "#F39B7F"}
    for ds in ("openstack", "aosp")
}

# モデルの表示名・色 (IRL=提案手法、RF=比較手法、Ens=アンサンブル)
MODEL_LABELS = {"cold_irl": "IRL（提案）", "rf_dir": "RF（比較）", "ens": "アンサンブル"}
MODEL_COLORS = {"cold_irl": "#E64B35", "rf_dir": "#4DBBD5", "ens": "#7E6148"}

# 差分ヒートマップ用 diverging (青=他原型より低い / 灰=中立 / 赤=他原型より高い)
DIVERGING = LinearSegmentedColormap.from_list(
    "arch_div",
    ["#17607A", "#4DBBD5", "#ECECEC", "#E64B35", "#8C2A1B"],
)

HORIZON_LABELS = {"0-3": "1–3m", "3-6": "4–6m", "6-9": "7–9m", "9-12": "10–12m"}
HORIZONS = ["0-3", "3-6", "6-9", "9-12"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 特徴量の並び順・短縮名 (archetype_profile_eugster.py:51-112 から移植)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEATURE_GROUPS = [
    ("在籍・量", [
        "window_tenure_days", "total_changes", "total_reviews",
        "active_months_ratio", "n_projects",
    ]),
    ("直近活動", [
        "recent_activity_frequency", "avg_activity_gap",
        "days_since_last_activity", "recent_load_ratio_30d_all",
        "recent_load_ratio_7d_30d", "concurrent_open_reviews_7d",
        "activity_trend",
    ]),
    ("承諾傾向", [
        "overall_acceptance_rate", "recent_acceptance_rate",
        "acceptance_rate_last10", "accept_rate_drift_30d",
        "time_since_last_accept", "time_since_last_reject",
        "recent_switch_rate",
    ]),
    ("役割・ネットワーク", [
        "reviewer_owner_ratio", "core_reviewer_ratio",
        "unique_collaborator_count", "reciprocity_score",
        "cross_project_review_share", "same_domain_share",
        "response_time_trend",
    ]),
    ("ディレクトリ親和度", [
        "path_review_count", "path_recency", "path_acceptance_rate",
        "path_owner_overlap", "path_hub_score", "path_neighbor_coverage",
    ]),
]
GROUP_OF = {f: g for g, feats in FEATURE_GROUPS for f in feats}

_SHORT = {
    "window_tenure_days": "tenure_days",
    "recent_activity_frequency": "act_freq",
    "recent_acceptance_rate": "accept_rate_rec",
    "overall_acceptance_rate": "accept_rate_all",
    "acceptance_rate_last10": "accept_last10",
    "recent_load_ratio_30d_all": "load_30d/all",
    "recent_load_ratio_7d_30d": "load_7d/30d",
    "days_since_last_activity": "days_inactive",
    "unique_collaborator_count": "n_collab",
    "concurrent_open_reviews_7d": "open_reviews_7d",
    "cross_project_review_share": "cross_proj",
    "reciprocity_score": "reciprocity",
    "reviewer_owner_ratio": "rev/owner",
    "core_reviewer_ratio": "core_rev",
    "time_since_last_accept": "tsl_accept",
    "time_since_last_reject": "tsl_reject",
    "accept_rate_drift_30d": "accept_drift",
    "recent_switch_rate": "switch_rate",
    "avg_activity_gap": "act_gap",
    "activity_trend": "act_trend",
    "active_months_ratio": "active_mo",
    "response_time_trend": "resp_trend",
    "same_domain_share": "same_domain",
    "n_projects": "n_proj",
    "total_changes": "n_changes",
    "total_reviews": "n_reviews",
    "path_review_count": "path_n_rev",
    "path_recency": "path_recency",
    "path_acceptance_rate": "path_acc_rate",
    "path_owner_overlap": "path_owner",
    "path_hub_score": "path_hub",
    "path_neighbor_coverage": "path_neigh",
}


def shorten(name: str) -> str:
    return _SHORT.get(name, name)


def ordered_features(columns) -> list[str]:
    """FEATURE_GROUPS の順に並べ替える。未定義の列は末尾に元の順で残す。"""
    cols = list(columns)
    ordered = [f for _, feats in FEATURE_GROUPS for f in feats if f in cols]
    return ordered + [c for c in cols if c not in ordered]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データセット定義 (4系統)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass(frozen=True)
class Source:
    dataset: str          # openstack / aosp
    level: str            # dev / pair
    dirpath: pathlib.Path
    alpha_file: str       # k{K}_*_alpha_with_metrics.csv のテンプレ
    hard_file: str
    pooled_file: str
    wilcoxon_file: str = "k{k}_wilcoxon.csv"
    profile_file: str = "k{k}_archetype_profile.csv"
    elbow_file: str = "rss_elbow.csv"
    n_col: str = "n_dev"
    prefix: str = "A"     # 原型ラベルの接頭辞 (dev=A, pair=P)
    names: dict = field(default_factory=dict)
    colors: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        proj = "OpenStack" if self.dataset == "openstack" else "AOSP"
        lvl = "開発者単位" if self.level == "dev" else "ペア単位"
        return f"{proj}・{lvl}"

    def path(self, template: str, k: int = 4) -> pathlib.Path:
        return self.dirpath / template.format(k=k)

    def rel(self, template: str, k: int = 4) -> str:
        return str(self.path(template, k).relative_to(ROOT))

    def arch_ids(self, k: int = 4) -> list[str]:
        return [f"{self.prefix}{i + 1}" for i in range(k)]

    def label(self, arch: str) -> str:
        """A1 / P1 → 'P1 他プロジェクト経験者'。"""
        key = f"{self.prefix}{arch[1:]}"
        return f"{key} {self.names.get(key, '')}".strip()

    def color(self, arch: str) -> str:
        return self.colors[f"{self.prefix}{arch[1:]}"]


def _src(dataset: str, level: str) -> Source:
    if dataset == "openstack":
        base = ROOT / "outputs" / "analysis"
        d = base / ("archetype_irl_rf_volat_ep150_coldonly" if level == "dev"
                    else "archetype_pair_level_volat_ep150_coldonly")
    else:
        d = ROOT / "outputs" / "aosp32" / "archetype" / level
    if level == "dev":
        return Source(
            dataset=dataset, level=level, dirpath=d,
            alpha_file="k{k}_dev_alpha_with_metrics.csv",
            hard_file="k{k}_hard_archetype_metrics.csv",
            pooled_file="k{k}_pooled_auc_bootstrap.csv",
            n_col="n_dev", prefix="A",
            names=DEV_NAMES[dataset], colors=DEV_COLORS[dataset],
        )
    return Source(
        dataset=dataset, level=level, dirpath=d,
        alpha_file="k{k}_pair_alpha_with_metrics.csv",
        hard_file="k{k}_hard_metrics.csv",
        pooled_file="k{k}_pooled_auc.csv",
        n_col="n_pair", prefix="P",
        names=PAIR_NAMES[dataset], colors=PAIR_COLORS[dataset],
    )


SOURCES = {(ds, lv): _src(ds, lv)
           for ds in ("openstack", "aosp") for lv in ("dev", "pair")}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 図のスタイルと保存
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE_W, SLIDE_H = 12.8, 6.6   # inch (16:8.25、pptx の本文領域に合わせる)


def use_slide_style() -> None:
    mpl.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 19,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.9,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def new_fig(w: float = SLIDE_W, h: float = SLIDE_H):
    use_slide_style()
    return plt.figure(figsize=(w, h))


_MANIFEST: list[dict] = []


def save_fig(fig, fig_id: str, title: str, message: str, source: str) -> pathlib.Path:
    """PDF と PNG を出力し、出所フッタを描く。manifest にも登録する。"""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.text(0.005, 0.006, f"出所: {source}", fontsize=8, color="#888888", ha="left")
    pdf = FIG / f"{fig_id}.pdf"
    png = FIG / f"{fig_id}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    _MANIFEST.append({
        "id": fig_id, "kind": "figure", "title": title,
        "message": message, "source": source,
        "path": str(png.relative_to(ROOT)),
    })
    print(f"[fig] {fig_id}: {png.relative_to(ROOT)}")
    return png


def save_table(df, fig_id: str, title: str, message: str, source: str,
               col_widths: list[float] | None = None, font_size: float = 11.0,
               index: bool = False) -> pathlib.Path:
    """表を CSV + Markdown で出力し、manifest に登録する (pptx 側でネイティブ表にする)。"""
    TAB.mkdir(parents=True, exist_ok=True)
    csv = TAB / f"{fig_id}.csv"
    df.to_csv(csv, index=index)
    md = TAB / f"{fig_id}.md"
    md.write_text(f"# {title}\n\n{message}\n\n" + df.to_markdown(index=index)
                  + f"\n\n出所: {source}\n", encoding="utf-8")
    _MANIFEST.append({
        "id": fig_id, "kind": "table", "title": title,
        "message": message, "source": source,
        "path": str(csv.relative_to(ROOT)),
        "col_widths": col_widths, "font_size": font_size,
    })
    print(f"[tab] {fig_id}: {csv.relative_to(ROOT)}")
    return csv


def save_section(fig_id: str, title: str, message: str) -> None:
    """章扉スライド。"""
    _MANIFEST.append({"id": fig_id, "kind": "section",
                      "title": title, "message": message, "source": ""})


def write_manifest(group: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    p = TAB / f"_manifest_{group}.json"
    p.write_text(json.dumps(_MANIFEST, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] {p.relative_to(ROOT)} ({len(_MANIFEST)} entries)")

"""phase2 共通ユーティリティ。

phase2 は「パイロットで得た相関(H1b支持)を、トップ国際学会水準の主張に耐えるところまで
詰める」フェーズ。既存の pilot/interim, pilot/processed のデータをそのまま入力に使い、
出力は pilot/phase2/outputs 以下に閉じる(既存の loop* の成果物は壊さない)。
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

REPO_ROOT = Path(__file__).resolve().parents[3]
# 期間を延ばした再収集(pilot/raw_full)に対しても同じ分析を回せるよう、
# 入出力ディレクトリは環境変数で差し替えられるようにする。既定値は phase1 と同じ。
#   PILOT_RAW=raw_full PILOT_INTERIM=interim_full PILOT_PROCESSED=processed_full PHASE2_OUT=outputs_full
RAW_DIR = REPO_ROOT / "pilot" / os.environ.get("PILOT_RAW", "raw")
INTERIM_DIR = REPO_ROOT / "pilot" / os.environ.get("PILOT_INTERIM", "interim")
PROCESSED_DIR = REPO_ROOT / "pilot" / os.environ.get("PILOT_PROCESSED", "processed")
CONFIG_DIR = REPO_ROOT / "pilot" / "config"
PHASE2_DIR = REPO_ROOT / "pilot" / "phase2"
OUT_DIR = PHASE2_DIR / os.environ.get("PHASE2_OUT", "outputs")
# 観測期間がこれより短いリポジトリは分析から除外する(収集が途中で止まったもの対策)。
# 「イベントがある月」で数えるので、活動が薄い月がある普通のリポジトリを落とさないよう低めにする。
# 収集が失敗したものは数ヶ月しか取れないので、これで区別できる。
MIN_OBS_MONTHS = int(os.environ.get("MIN_OBS_MONTHS", "6"))

FIRST_BATCH_REPOS = [
    "home-assistant/core",
    "grafana/grafana",
    "juspay/hyperswitch",
    "ray-project/ray",
    "python/cpython",
    "ghostty-org/ghostty",
    "apache/airflow",
    "godotengine/godot",
]


def load_repos() -> list[str]:
    repos = list(FIRST_BATCH_REPOS)
    batch2_csv = CONFIG_DIR / "loop7_selected_repos_batch2_final.csv"
    if batch2_csv.exists():
        with batch2_csv.open() as f:
            repos += [row["full_name"] for row in csv.DictReader(f)]
    return repos


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def build_count_panel() -> pd.DataFrame:
    """module x month の「工程別・担い手カテゴリ別のイベント件数」パネルを全リポジトリ分作る。

    H_E のような比率ではなく件数(絶対量)を持つのがポイント。
    比率の低下が「人間の退出(分子の減少)」なのか「AIの上乗せ(分母の膨張)」なのかを
    分解するために必要になる。

    legacy_bot は主解析と同じく除外する(研究設計書 §3.1)。
    agent 判定は trailer 込み(gates.md Loop1 で確定した主指標)。
    """
    frames = []
    for repo in load_repos():
        ev_path = INTERIM_DIR / repo_dir_name(repo) / "events.parquet"
        if not ev_path.exists():
            continue
        ev = pd.read_parquet(ev_path)
        ev = ev[ev["stage"].isin(["G", "E"]) & (ev["actor_category"] != "legacy_bot")].copy()
        if ev.empty:
            continue
        ev["is_human"] = ev["actor_category"] == "human"
        trailer = ev["trailer_agent_detected"].fillna(False).astype(bool)
        # 人間作成でも commit trailer に AI エージェントが残っていれば agent 側に数える
        ev["is_human_strict"] = ev["is_human"] & ~trailer

        g = ev[ev["stage"] == "G"]
        e = ev[ev["stage"] == "E"]
        key = ["module_id", "month"]

        gg = g.groupby(key).agg(
            n_G=("is_human", "size"),
            n_G_human=("is_human", "sum"),
            n_G_human_strict=("is_human_strict", "sum"),
            n_G_actors=("actor_login", "nunique"),
        )
        ee = e.groupby(key).agg(
            n_E=("is_human", "size"),
            n_E_human=("is_human", "sum"),
            n_E_actors=("actor_login", "nunique"),
        )
        e_h_actors = (
            e[e["is_human"]].groupby(key)["actor_login"].nunique().rename("n_E_human_actors")
        )
        g_h_actors = (
            g[g["is_human_strict"]].groupby(key)["actor_login"].nunique().rename("n_G_human_actors")
        )
        # 評価対象になった PR の数(レビュー件数ではなく「レビューされた PR 数」)
        e_prs = e.groupby(key)["pr_number"].nunique().rename("n_E_prs")
        g_prs = g.groupby(key)["pr_number"].nunique().rename("n_G_prs")

        panel = gg.join(ee, how="outer").join(e_h_actors).join(g_h_actors).join(e_prs).join(g_prs)
        panel = panel.reset_index()
        panel["repo"] = repo
        frames.append(panel)

    df = pd.concat(frames, ignore_index=True)

    # 収集が途中で止まったリポジトリ(GitHub 側の INTERNAL エラーで遡れなかった等)は
    # 観測期間が極端に短くなる。そのまま混ぜるとイベントスタディの対照群を歪めるので落とす。
    # 13ヶ月版は全リポジトリが13ヶ月あるので、この閾値では何も落ちない。
    months_per_repo = df.groupby("repo")["month"].nunique()
    too_short = months_per_repo[months_per_repo < MIN_OBS_MONTHS]
    if len(too_short):
        print(f"  [warn] 観測月数が {MIN_OBS_MONTHS} 未満のため除外: "
              + ", ".join(f"{r}({n}ヶ月)" for r, n in too_short.items()))
        df = df[~df["repo"].isin(too_short.index)].copy()

    # イベントが1件も無い工程は「0件」として扱う(観測はされている月なので欠測ではない)
    count_cols = [c for c in df.columns if c.startswith("n_")]
    df[count_cols] = df[count_cols].fillna(0.0)

    df["n_G_agent"] = df["n_G"] - df["n_G_human_strict"]
    df["n_E_agent"] = df["n_E"] - df["n_E_human"]
    # 主指標: 生成工程の AI 関与度 A'(m,t)。G イベントが無い月は定義できない
    df["A_prime"] = np.where(df["n_G"] > 0, df["n_G_agent"] / df["n_G"], np.nan)
    df["H_E"] = np.where(df["n_E"] > 0, df["n_E_human"] / df["n_E"], np.nan)
    df["H_G_prime"] = np.where(df["n_G"] > 0, df["n_G_human_strict"] / df["n_G"], np.nan)

    df["module_id_global"] = df["repo"] + "::" + df["module_id"]
    df["month_period"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
    return df


def fit_panel(
    df: pd.DataFrame,
    y: str,
    x_cols: list[str],
    *,
    entity_effects: bool = True,
    time_effects: bool = True,
) -> dict:
    """entity(module) + time 固定効果の PanelOLS を推定し、module / repo 両方のクラスタ標準誤差を返す。

    df は module_id_global, month_period, repo 列を持つこと。
    """
    need = [y, *x_cols]
    d = df.dropna(subset=need).copy()
    d = d.set_index(["module_id_global", "month_period"])
    endog = d[y]
    exog = d[x_cols]

    mod = PanelOLS(endog, exog, entity_effects=entity_effects, time_effects=time_effects, drop_absorbed=True)
    res_m = mod.fit(cov_type="clustered", cluster_entity=True)

    repo_series = d["repo"]
    res_r = PanelOLS(
        endog, exog, entity_effects=entity_effects, time_effects=time_effects, drop_absorbed=True
    ).fit(cov_type="clustered", clusters=repo_series)

    return {
        "y": y,
        "x": x_cols,
        "n_obs": int(res_m.nobs),
        "n_modules": int(d.index.get_level_values(0).nunique()),
        "n_repos": int(repo_series.nunique()),
        "params": res_m.params.to_dict(),
        "se_module": res_m.std_errors.to_dict(),
        "p_module": res_m.pvalues.to_dict(),
        "se_repo": res_r.std_errors.to_dict(),
        "p_repo": res_r.pvalues.to_dict(),
        "rsquared_within": float(res_m.rsquared_within),
    }


def fmt_result(r: dict, key: str | None = None) -> str:
    """fit_panel の結果を1行に要約する(主要な説明変数のみ)。"""
    k = key or r["x"][0]
    return (
        f"{r['y']:<28} {k}={r['params'][k]:+.4f} "
        f"p_mod={r['p_module'][k]:.4g} p_repo={r['p_repo'][k]:.4g} "
        f"(n={r['n_obs']}, modules={r['n_modules']}, repos={r['n_repos']})"
    )

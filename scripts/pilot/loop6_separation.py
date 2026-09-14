"""Loop 6 — 分離度4軸(S1〜S4)の算出と相互相関。

S1 差分型: H_E-H_G, H_I-H_G (panelから直接)
S2 集合重なり型: Jaccard(担い手集合_G, 担い手集合_E) (module×monthごと)
S3 個人役割移動型: actorごとの reviewer_ratio = n_review/(n_review+n_pr) の時系列トレンド
S4 時間的先行型: module×monthのH_G, H_E系列の相互相関(ラグ0〜3ヶ月)、最大相関を与えるラグの分布

実行: uv run python scripts/pilot/loop6_separation.py <owner>/<repo>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = REPO_ROOT / "pilot" / "interim"
PROCESSED_DIR = REPO_ROOT / "pilot" / "processed"


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def compute_s1(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["S1_EG"] = out["H_E"] - out["H_G"]
    out["S1_IG"] = out["H_I"] - out["H_G"]
    if "H_G_prime" in out.columns:
        # H_G は PR作成者(actor login)のみの判定で、commit trailerのagent関与を見落とすため過大評価されがち。
        # H_G_prime(trailer込み)を使った版も併記する(Loop4の発見: A と A_prime で大きく値が異なる)
        out["S1_EG_prime"] = out["H_E"] - out["H_G_prime"]
        out["S1_IG_prime"] = out["H_I"] - out["H_G_prime"]
    return out


def compute_s2(events: pd.DataFrame) -> pd.DataFrame:
    """S2: Jaccard(担い手集合_G, 担い手集合_E)。

    trailer込み版は今回見送った: G側にtrailerで検出したエージェント名(Claude/Copilot等)を
    独立の主体として加えても、E側の実在アカウント(copilot-pull-request-reviewer等)と
    名前が一致せず分子に寄与しない(分母だけ増えてJaccardが一律下がるだけの指標になる)。
    意味のある拡張には trailer名 <-> E側login名 の名寄せ表が要る。時間の制約で見送り、Loop8までの持ち越しとする。
    """
    rows = []
    for (mod, month), grp in events.groupby(["module_id", "month"]):
        g_actors = set(grp.loc[grp["stage"] == "G", "actor_login"])
        e_actors = set(grp.loc[grp["stage"] == "E", "actor_login"])
        union = g_actors | e_actors
        jaccard = len(g_actors & e_actors) / len(union) if union else np.nan
        rows.append({"module_id": mod, "month": month, "S2_jaccard": jaccard, "n_g_actors": len(g_actors), "n_e_actors": len(e_actors)})
    return pd.DataFrame(rows)


def compute_s3(events: pd.DataFrame) -> pd.DataFrame:
    ge = events[events["stage"].isin(["G", "E"])]
    rows = []
    for (actor, month), grp in ge.groupby(["actor_login", "month"]):
        n_pr = (grp["stage"] == "G").sum()
        n_review = (grp["stage"] == "E").sum()
        total = n_pr + n_review
        if total == 0:
            continue
        rows.append({"actor_login": actor, "month": month, "reviewer_ratio": n_review / total, "n_total": total})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # actorごとに月順で並べ、最初と最後の reviewer_ratio の差(単純トレンド)を計算
    trends = []
    for actor, grp in df.groupby("actor_login"):
        grp = grp.sort_values("month")
        if len(grp) < 2:
            continue
        delta = grp["reviewer_ratio"].iloc[-1] - grp["reviewer_ratio"].iloc[0]
        trends.append({"actor_login": actor, "n_months_active": len(grp), "reviewer_ratio_delta": delta})
    return pd.DataFrame(trends)


def compute_s4(panel: pd.DataFrame, g_col: str = "H_G", e_col: str = "H_E", max_lag: int = 3) -> pd.DataFrame:
    rows = []
    for mod, grp in panel.groupby("module_id"):
        grp = grp.sort_values("month")
        g = grp[g_col].to_numpy()
        e = grp[e_col].to_numpy()
        valid = ~(np.isnan(g) | np.isnan(e))
        if valid.sum() < 6:  # 系列が短すぎると変化点/相関の意味が薄い
            continue
        g, e = g[valid], e[valid]
        best_lag, best_corr = None, -2.0
        for lag in range(0, max_lag + 1):
            if lag == 0:
                gg, ee = g, e
            else:
                gg, ee = g[:-lag], e[lag:]
            if len(gg) < 4 or np.std(gg) == 0 or np.std(ee) == 0:
                continue
            corr = np.corrcoef(gg, ee)[0, 1]
            if corr > best_corr:
                best_corr, best_lag = corr, lag
        if best_lag is not None:
            rows.append({"module_id": mod, "best_lag_months": best_lag, "best_corr": best_corr, "n_months": len(g)})
    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: loop6_separation.py <owner>/<repo>")
        sys.exit(1)
    repo = sys.argv[1]
    interim_d = INTERIM_DIR / repo_dir_name(repo)
    processed_d = PROCESSED_DIR / repo_dir_name(repo)

    panel = pd.read_parquet(processed_d / "panel.parquet")
    events = pd.read_parquet(interim_d / "events.parquet")

    panel_s1 = compute_s1(panel)
    s2 = compute_s2(events)
    s3 = compute_s3(events)
    s4 = compute_s4(panel_s1, "H_G", "H_E")
    s4_prime = compute_s4(panel_s1, "H_G_prime", "H_E") if "H_G_prime" in panel_s1.columns else pd.DataFrame()

    merged = panel_s1.merge(s2, on=["module_id", "month"], how="left")
    merged.to_parquet(processed_d / "separation_axes.parquet", index=False)
    s3.to_parquet(processed_d / "s3_role_shift.parquet", index=False)
    s4.to_parquet(processed_d / "s4_lead_lag.parquet", index=False)
    if len(s4_prime):
        s4_prime.to_parquet(processed_d / "s4_lead_lag_prime.parquet", index=False)

    corr_cols = [c for c in ["S1_EG", "S1_IG", "S1_EG_prime", "S1_IG_prime", "S2_jaccard"] if c in merged.columns]
    corr_matrix = merged[corr_cols].corr(numeric_only=True)

    summary = {
        "repo": repo,
        "S1_EG_mean": float(merged["S1_EG"].mean(skipna=True)),
        "S1_IG_mean": float(merged["S1_IG"].mean(skipna=True)),
        "S1_EG_prime_mean(trailer込み)": float(merged["S1_EG_prime"].mean(skipna=True)) if "S1_EG_prime" in merged.columns else None,
        "S1_IG_prime_mean(trailer込み)": float(merged["S1_IG_prime"].mean(skipna=True)) if "S1_IG_prime" in merged.columns else None,
        "S2_jaccard_mean": float(merged["S2_jaccard"].mean(skipna=True)),
        "S3_n_actors_with_trend": len(s3),
        "S3_reviewer_ratio_delta_mean": float(s3["reviewer_ratio_delta"].mean()) if len(s3) else None,
        "S4_n_modules_computed": len(s4),
        "S4_lag_distribution": s4["best_lag_months"].value_counts().to_dict() if len(s4) else {},
        "S4_prime_n_modules_computed(trailer込み)": len(s4_prime),
        "S4_prime_lag_distribution(trailer込み)": s4_prime["best_lag_months"].value_counts().to_dict() if len(s4_prime) else {},
        "S1_S2_corr_matrix": corr_matrix.to_dict(),
    }
    (processed_d / "loop6_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"=== {repo} ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

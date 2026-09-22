"""D14: 用量反応 — AI がもっと大量に入れば人間は減るのか。

D4/D7/D8 の処置は二値(AI レビューを導入したか)だった。しかし導入後の AI レビュー比率は
リポジトリによって大きく違う(backstage は 40%超、grafana は 25% 程度)。
「平均的には効果ゼロ」でも「大量に入ったところでは効果がある」かもしれない。

これは null result の最も重要な頑健性チェックでもある。
高用量でも人間が減らないなら「AI は人間を代替しない」という主張はかなり強くなる。
逆に高用量で減るなら、それ自体が新しい発見になる。

3通りで見る:
  A. 連続処置 DiD: intensity_r(導入後の AI レビュー比率) x post のクロス項
  B. 高用量 / 低用量に分けたイベントスタディ
  C. モジュールレベルの用量(内生的なので探索的)

実行: uv run python pilot/phase2/scripts/d14_dose_response.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from common import OUT_DIR, build_count_panel
from d4_repo_event_study import LEAD_LAG, REF, build_balanced, detect_adoption

OUTCOMES = ["log_n_E_human", "log_n_E_agent", "H_E", "e_human_per_pr", "log_n_E_human_actors"]


def add_outcomes(b: pd.DataFrame) -> pd.DataFrame:
    b["log_n_E_human"] = np.log1p(b["n_E_human"])
    b["log_n_E_agent"] = np.log1p(b["n_E_agent"])
    b["log_n_E_human_actors"] = np.log1p(b["n_E_human_actors"])
    b["H_E"] = np.where(b["n_E"] > 0, b["n_E_human"] / b["n_E"], np.nan)
    b["e_human_per_pr"] = np.where(b["n_E_prs"] > 0, b["n_E_human"] / b["n_E_prs"], np.nan)
    return b


def fit(d: pd.DataFrame, y: str, cols: list[str]) -> dict:
    dd = d.dropna(subset=[y, *cols]).set_index(["module_id_global", "month_period"])
    r = PanelOLS(dd[y], dd[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", clusters=dd["repo"]
    )
    return {"params": r.params.to_dict(), "p_repo": r.pvalues.to_dict(),
            "se_repo": r.std_errors.to_dict(), "n_obs": int(r.nobs)}


def main() -> None:
    df = build_count_panel()
    months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(months)}
    adoption, never, always = detect_adoption(df)

    # 各処置リポジトリの「導入後の AI レビュー比率」= 用量
    by = df.groupby(["repo", "month"])[["n_E_agent", "n_E_human"]].sum().reset_index()
    by["share"] = by["n_E_agent"] / (by["n_E_agent"] + by["n_E_human"])
    intensity = {}
    for repo, gm in adoption.items():
        s = by[(by["repo"] == repo) & (by["month"] >= gm)]["share"]
        intensity[repo] = float(s.mean())

    print("=== 処置リポジトリの用量(導入後の平均 AI レビュー比率) ===")
    for r, v in sorted(intensity.items(), key=lambda kv: -kv[1]):
        print(f"  {v:6.1%}  {r}  (導入 {adoption[r]})")
    med = float(np.median(list(intensity.values())))
    print(f"  中央値 {med:.1%}")

    bal = add_outcomes(build_balanced(df))
    bal = bal[bal["repo"].isin(set(adoption) | set(never))].copy()
    bal["g"] = bal["repo"].map({r: midx[m] for r, m in adoption.items()})
    bal["post"] = ((bal["month_idx"] >= bal["g"]) & bal["g"].notna()).astype(float)
    bal["intensity"] = bal["repo"].map(intensity).fillna(0.0)
    bal["dose"] = bal["post"] * bal["intensity"]

    out: dict = {"intensity": intensity, "median_intensity": med, "A_continuous": {}, "B_split": {}}

    # --- A. 連続処置 DiD --------------------------------------------------------
    print("\n=== A. 連続処置 DiD(dose = post x 導入後 AI レビュー比率) ===")
    print("  dose の係数 = 「AI レビュー比率が 0 から 1 に増えたときの効果」")
    print(f"  {'被説明変数':<24}{'dose係数':>12}{'p(repo)':>10}{'用量中央値での効果':>20}")
    for y in OUTCOMES:
        r = fit(bal, y, ["dose"])
        b = r["params"]["dose"]
        out["A_continuous"][y] = r
        print(f"  {y:<24}{b:>+12.4f}{r['p_repo']['dose']:>10.4g}{b * med:>20.4f}")

    # --- B. 高用量 / 低用量に分けたイベントスタディ --------------------------------
    print("\n=== B. 高用量 / 低用量に分けたイベントスタディ ===")
    for label, repos in [
        ("高用量", [r for r, v in intensity.items() if v >= med]),
        ("低用量", [r for r, v in intensity.items() if v < med]),
    ]:
        sub = bal[bal["repo"].isin(set(repos) | set(never))].copy()
        ev_cols = []
        for j in range(-LEAD_LAG, LEAD_LAG + 1):
            if j == REF:
                continue
            name = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
            k = sub["month_idx"] - sub["g"]
            val = (k <= j) if j == -LEAD_LAG else ((k >= j) if j == LEAD_LAG else (k == j))
            sub[name] = val.fillna(False).astype(float)
            ev_cols.append(name)
        print(f"\n  [{label}] {len(repos)} リポジトリ: {', '.join(sorted(repos))}")
        print(f"  {'被説明変数':<24}{'k=+2':>12}{'p':>9}{'k=+3':>12}{'p':>9}")
        out["B_split"][label] = {}
        for y in OUTCOMES:
            r = fit(sub, y, ev_cols)
            out["B_split"][label][y] = r
            b2, p2 = r["params"].get("k_p2", np.nan), r["p_repo"].get("k_p2", np.nan)
            b3, p3 = r["params"].get("k_p3", np.nan), r["p_repo"].get("k_p3", np.nan)
            print(f"  {y:<24}{b2:>+12.4f}{p2:>9.3f}{b3:>+12.4f}{p3:>9.3f}")

    # --- C. モジュールレベルの用量(探索的) ----------------------------------------
    # 「そのモジュールに AI レビューがどれだけ集中したか」で層別する。
    # モジュールへの AI の入り方は内生的なので因果とは解釈できない
    print("\n=== C. モジュールレベルの用量(探索的、内生性あり) ===")
    mod_dose = (
        df.groupby("module_id_global")
        .apply(lambda g: g["n_E_agent"].sum() / max(g["n_E_agent"].sum() + g["n_E_human"].sum(), 1),
               include_groups=False)
        .rename("mod_intensity")
    )
    bal2 = bal.join(mod_dose, on="module_id_global")
    qs = bal2["mod_intensity"].quantile([0.5, 0.75, 0.9])
    print(f"  モジュール AI レビュー比率の分位点: 50%={qs[0.5]:.3f} 75%={qs[0.75]:.3f} 90%={qs[0.9]:.3f}")
    out["C_module_dose"] = {}
    for label, lo in [("上位25%", qs[0.75]), ("上位10%", qs[0.9])]:
        sub = bal2[(bal2["mod_intensity"] >= lo) | bal2["repo"].isin(never)].copy()
        ev_cols = []
        for j in range(-LEAD_LAG, LEAD_LAG + 1):
            if j == REF:
                continue
            name = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
            k = sub["month_idx"] - sub["g"]
            val = (k <= j) if j == -LEAD_LAG else ((k >= j) if j == LEAD_LAG else (k == j))
            sub[name] = val.fillna(False).astype(float)
            ev_cols.append(name)
        print(f"\n  [{label}] n_modules={sub['module_id_global'].nunique():,}")
        print(f"  {'被説明変数':<24}{'k=+2':>12}{'p':>9}{'k=+3':>12}{'p':>9}")
        out["C_module_dose"][label] = {}
        for y in ["log_n_E_human", "H_E", "e_human_per_pr"]:
            try:
                r = fit(sub, y, ev_cols)
            except Exception as e:
                print(f"  {y:<24} 推定できず ({type(e).__name__})")
                continue
            out["C_module_dose"][label][y] = r
            b2, p2 = r["params"].get("k_p2", np.nan), r["p_repo"].get("k_p2", np.nan)
            b3, p3 = r["params"].get("k_p3", np.nan), r["p_repo"].get("k_p3", np.nan)
            print(f"  {y:<24}{b2:>+12.4f}{p2:>9.3f}{b3:>+12.4f}{p3:>9.3f}")

    (OUT_DIR / "d14_dose_response.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd14_dose_response.json'}")


if __name__ == "__main__":
    main()

"""D6: D4(リポジトリ単位イベントスタディ)の頑健性。

D4 の弱点は3つある。
  (1) 処置リポジトリが7件しかない。クラスタ標準誤差の漸近論が効きにくい
  (2) 導入の検出が閾値(JUMP=0.05, LEVEL=0.10)に依存している
  (3) 2WFE の staggered DiD は、先に処置された群が後の群の対照になることで負の重みを持ちうる

対処:
  A. 閾値の感度分析 — JUMP/LEVEL を動かして simple DiD の符号と大きさが安定するか
  B. stacked DiD — 各処置コホートを never-treated とだけ比較し、積み上げて推定
     (Cengiz et al. 2019)。既処置を対照に使わないので負の重み問題が生じない
  C. randomization inference — 処置の割り当てをランダムに入れ替えた偽処置を多数作り、
     実際の推定値が偽処置の分布のどこに位置するかで p 値を出す。
     処置群が少ないときにクラスタ標準誤差より信頼できる

実行: uv run python pilot/phase2/scripts/d6_robustness.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from common import OUT_DIR, build_count_panel
from d4_repo_event_study import build_balanced, detect_adoption

WIN = 3
OUTCOMES = ["log_n_E_agent", "log_n_E_human", "H_E", "e_human_per_pr"]
N_PERM = 300
RNG = np.random.default_rng(20260921)


def add_outcomes(bal: pd.DataFrame) -> pd.DataFrame:
    bal["log_n_E_human"] = np.log1p(bal["n_E_human"])
    bal["log_n_E_agent"] = np.log1p(bal["n_E_agent"])
    bal["H_E"] = np.where(bal["n_E"] > 0, bal["n_E_human"] / bal["n_E"], np.nan)
    bal["e_human_per_pr"] = np.where(bal["n_E_prs"] > 0, bal["n_E_human"] / bal["n_E_prs"], np.nan)
    return bal


def simple_did(bal: pd.DataFrame, y: str, treat_col: str = "post_treat") -> dict:
    """y ~ post_treat + module_FE + month_FE。係数1つだけの素朴な DiD。"""
    d = bal.dropna(subset=[y]).set_index(["module_id_global", "month_period"])
    res = PanelOLS(d[y], d[[treat_col]], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", clusters=d["repo"]
    )
    return {
        "coef": float(res.params[treat_col]),
        "se_repo": float(res.std_errors[treat_col]),
        "p_repo": float(res.pvalues[treat_col]),
        "n_obs": int(res.nobs),
    }


def coef_only(bal: pd.DataFrame, y: str, treat_col: str) -> float:
    """置換検定用。標準誤差を計算しない分だけ速い。"""
    d = bal.dropna(subset=[y]).set_index(["module_id_global", "month_period"])
    res = PanelOLS(d[y], d[[treat_col]], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="unadjusted"
    )
    return float(res.params[treat_col])


def main() -> None:
    df = build_count_panel()
    months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(months)}
    out: dict = {}

    # =========================================================================
    # A. 閾値の感度分析
    # =========================================================================
    print("=== A. 閾値の感度分析(simple DiD) ===")
    sens = []
    for jump in [0.03, 0.05, 0.08, 0.10]:
        for level in [0.05, 0.10, 0.15]:
            import d4_repo_event_study as d4

            d4.JUMP, d4.LEVEL = jump, level
            adoption, never, always = detect_adoption(df)
            if len(adoption) < 3:
                continue
            bal = build_balanced(df)
            bal = bal[bal["repo"].isin(set(adoption) | set(never))].copy()
            bal = add_outcomes(bal)
            bal["g"] = bal["repo"].map({r: midx[m] for r, m in adoption.items()})
            bal["post_treat"] = ((bal["month_idx"] >= bal["g"]) & bal["g"].notna()).astype(float)
            row = {"jump": jump, "level": level, "n_treated": len(adoption), "n_never": len(never)}
            for y in OUTCOMES:
                r = simple_did(bal, y)
                row[f"{y}_coef"] = r["coef"]
                row[f"{y}_p"] = r["p_repo"]
            sens.append(row)
            print(f"  jump={jump:.2f} level={level:.2f} treated={len(adoption):2d} | "
                  + " ".join(f"{y.replace('log_n_E_','').replace('_','')}={row[f'{y}_coef']:+.3f}"
                             f"(p={row[f'{y}_p']:.3f})" for y in OUTCOMES))
    out["threshold_sensitivity"] = sens

    # 以降は主仕様 (JUMP=0.05, LEVEL=0.10) に戻す
    import d4_repo_event_study as d4

    d4.JUMP, d4.LEVEL = 0.05, 0.10
    adoption, never, always = detect_adoption(df)
    bal_all = build_balanced(df)
    bal_main = bal_all[bal_all["repo"].isin(set(adoption) | set(never))].copy()
    bal_main = add_outcomes(bal_main)
    bal_main["g"] = bal_main["repo"].map({r: midx[m] for r, m in adoption.items()})
    bal_main["post_treat"] = ((bal_main["month_idx"] >= bal_main["g"]) & bal_main["g"].notna()).astype(float)

    print("\n=== 主仕様の simple DiD ===")
    out["main_did"] = {}
    for y in OUTCOMES:
        r = simple_did(bal_main, y)
        out["main_did"][y] = r
        print(f"  {y:<18} {r['coef']:+.4f} (se={r['se_repo']:.4f}, p={r['p_repo']:.4g}, n={r['n_obs']})")

    # =========================================================================
    # B. stacked DiD — 各コホート x never-treated だけを比較して積み上げる
    # =========================================================================
    print("\n=== B. stacked DiD(対照は never-treated のみ) ===")
    stacks = []
    for repo, gm in adoption.items():
        gi = midx[gm]
        lo, hi = gi - WIN, gi + WIN
        sub = bal_all[
            bal_all["repo"].isin({repo} | set(never)) & bal_all["month_idx"].between(lo, hi)
        ].copy()
        if sub.empty:
            continue
        sub["stack"] = repo
        sub["post_treat"] = ((sub["repo"] == repo) & (sub["month_idx"] >= gi)).astype(float)
        stacks.append(sub)
    stacked = pd.concat(stacks, ignore_index=True)
    stacked = add_outcomes(stacked)
    # stack ごとに固定効果を分ける(同じモジュールが複数 stack に現れるため)
    stacked["module_stack"] = stacked["module_id_global"] + "@@" + stacked["stack"]
    stacked["month_stack"] = stacked["month"] + "@@" + stacked["stack"]

    out["stacked_did"] = {}
    for y in OUTCOMES:
        d = stacked.dropna(subset=[y]).copy()
        # month_stack を時間次元として使うため、ダミーで入れる代わりに
        # entity=module_stack, time=month_period とし、stack x month の交互作用を明示的に入れる
        d = d.set_index(["module_stack", "month_period"])
        dummies = pd.get_dummies(d["month_stack"], drop_first=True, dtype=float)
        X = pd.concat([d[["post_treat"]], dummies], axis=1)
        res = PanelOLS(d[y], X, entity_effects=True, drop_absorbed=True).fit(
            cov_type="clustered", clusters=d["repo"]
        )
        rec = {
            "coef": float(res.params["post_treat"]),
            "se_repo": float(res.std_errors["post_treat"]),
            "p_repo": float(res.pvalues["post_treat"]),
            "n_obs": int(res.nobs),
        }
        out["stacked_did"][y] = rec
        print(f"  {y:<18} {rec['coef']:+.4f} (se={rec['se_repo']:.4f}, p={rec['p_repo']:.4g}, n={rec['n_obs']})")

    # =========================================================================
    # C. randomization inference — 偽の処置割り当てを多数作る
    # =========================================================================
    print(f"\n=== C. randomization inference({N_PERM}回の偽処置) ===")
    pool = sorted(set(adoption) | set(never))
    real_gs = [midx[m] for m in adoption.values()]
    n_treat = len(adoption)

    out["randomization_inference"] = {}
    for y in OUTCOMES:
        actual = out["main_did"][y]["coef"]
        null_coefs = []
        for _ in range(N_PERM):
            fake_repos = RNG.choice(pool, size=n_treat, replace=False)
            fake_g = RNG.choice(real_gs, size=n_treat, replace=True)
            gmap = dict(zip(fake_repos, fake_g))
            b = bal_main.copy()
            b["g_fake"] = b["repo"].map(gmap)
            b["post_fake"] = ((b["month_idx"] >= b["g_fake"]) & b["g_fake"].notna()).astype(float)
            if b["post_fake"].sum() == 0:
                continue
            try:
                null_coefs.append(coef_only(b, y, "post_fake"))
            except Exception:
                continue
        null_coefs = np.array(null_coefs)
        p_ri = float((np.abs(null_coefs) >= abs(actual)).mean())
        rec = {
            "actual_coef": actual,
            "n_permutations": int(len(null_coefs)),
            "null_mean": float(null_coefs.mean()),
            "null_sd": float(null_coefs.std()),
            "null_q025": float(np.quantile(null_coefs, 0.025)),
            "null_q975": float(np.quantile(null_coefs, 0.975)),
            "p_randomization": p_ri,
        }
        out["randomization_inference"][y] = rec
        print(f"  {y:<18} actual={actual:+.4f}  null 95%CI=[{rec['null_q025']:+.4f}, {rec['null_q975']:+.4f}]  "
              f"p_RI={p_ri:.4f}")

    (OUT_DIR / "d6_robustness.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd6_robustness.json'}")


if __name__ == "__main__":
    main()

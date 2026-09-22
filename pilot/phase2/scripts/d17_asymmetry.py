"""D17: ★生成と評価の非対称性を直接検定する。

D16 で、同じ「AI 導入」イベントでも生成工程と評価工程で人間の反応が違う可能性が見えた。
  生成側: log(1+人間の生成件数) k=+3 で -0.091 (p=0.064)、「10%以上の減少なし」が言えない
  評価側: log(1+人間の評価件数) は 10%以上の減少を棄却できる (p=0.0013)

しかしこれは別々のモデルの結果を並べただけで、「差が有意か」は検定していない。
ここでは module x month x stage の縦持ちパネルを作り、

    log(1+人間の件数) ~ post + post x is_generation + (module x stage)FE + (month x stage)FE

として **post x is_generation の係数**を直接推定する。これが非対称性そのものであり、
本研究の元々の問い「生成と評価の分離は起きているか」に対する答えになる。

処置は2通りで確認する。
  (i) 生成側の導入イベント(コーディングエージェントによる PR 作成の開始)
  (ii) 評価側の導入イベント(AI レビューボットの有効化)

実行: uv run python pilot/phase2/scripts/d17_asymmetry.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from common import OUT_DIR, build_count_panel
from d4_repo_event_study import LEAD_LAG, REF, build_balanced, detect_adoption
from d16_generation_side import detect_generation_adoption


def build_long(bal: pd.DataFrame) -> pd.DataFrame:
    """module x month x stage(G/E)の縦持ちにする。"""
    g = bal[["module_id_global", "repo", "month", "month_idx", "month_period", "g"]].copy()
    g["stage"] = "G"
    g["n_human"] = bal["n_G_human_strict"].values
    g["n_agent"] = bal["n_G_agent"].values
    g["n_human_actors"] = bal["n_G_human_actors"].values

    e = bal[["module_id_global", "repo", "month", "month_idx", "month_period", "g"]].copy()
    e["stage"] = "E"
    e["n_human"] = bal["n_E_human"].values
    e["n_agent"] = bal["n_E_agent"].values
    e["n_human_actors"] = bal["n_E_human_actors"].values

    long = pd.concat([g, e], ignore_index=True)
    long["is_G"] = (long["stage"] == "G").astype(float)
    long["log_n_human"] = np.log1p(long["n_human"])
    long["log_n_agent"] = np.log1p(long["n_agent"])
    long["log_n_human_actors"] = np.log1p(long["n_human_actors"])
    long["entity"] = long["module_id_global"] + "@@" + long["stage"]
    long["month_stage"] = long["month"] + "@@" + long["stage"]
    return long


def run(long: pd.DataFrame, y: str, treat_cols: list[str], label: str) -> dict:
    """post(および post x is_G)を、(module x stage)FE と (month x stage)FE の下で推定する。"""
    d = long.dropna(subset=[y, *treat_cols]).copy()
    d = d.set_index(["entity", "month_period"])
    # month x stage の固定効果はダミーで明示的に入れる(time_effects は month のみになるため)
    dummies = pd.get_dummies(d["month_stage"], drop_first=True, dtype=float)
    X = pd.concat([d[treat_cols], dummies], axis=1)
    res = PanelOLS(d[y], X, entity_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", clusters=d["repo"]
    )
    return {
        "label": label, "y": y,
        "params": {c: float(res.params[c]) for c in treat_cols if c in res.params},
        "se_repo": {c: float(res.std_errors[c]) for c in treat_cols if c in res.params},
        "p_repo": {c: float(res.pvalues[c]) for c in treat_cols if c in res.params},
        "n_obs": int(res.nobs), "n_repos": int(d["repo"].nunique()),
    }


def main() -> None:
    df = build_count_panel()
    months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(months)}

    adopt_E, never_E, always_E = detect_adoption(df)
    adopt_G, never_G, always_G = detect_generation_adoption(df)

    out: dict = {"adoption_E": adopt_E, "adoption_G": adopt_G, "results": {}}

    for tag, adoption, never in [
        ("(i) 生成側の導入イベント", adopt_G, never_G),
        ("(ii) 評価側の導入イベント", adopt_E, never_E),
    ]:
        bal = build_balanced(df)
        bal = bal[bal["repo"].isin(set(adoption) | set(never))].copy()
        bal["g"] = bal["repo"].map({r: midx[m] for r, m in adoption.items()})
        long = build_long(bal)
        long["post"] = ((long["month_idx"] >= long["g"]) & long["g"].notna()).astype(float)
        long["post_G"] = long["post"] * long["is_G"]

        print(f"\n{'=' * 78}")
        print(f"=== {tag}  (treated {len(adoption)} / never {len(never)}) ===")
        print(f"{'=' * 78}")

        # 1) 単純な post(生成と評価をまとめた平均効果)
        r_pool = run(long, "log_n_human", ["post"], f"{tag} pooled")
        # 2) 非対称性: post + post x is_G
        r_asym = run(long, "log_n_human", ["post", "post_G"], f"{tag} asymmetry")
        # first stage の確認(AI 側が実際に増えたか)
        r_ai = run(long, "log_n_agent", ["post", "post_G"], f"{tag} agent")
        # 人数でも見る
        r_act = run(long, "log_n_human_actors", ["post", "post_G"], f"{tag} actors")

        out["results"][tag] = {"pooled": r_pool, "asymmetry": r_asym,
                               "agent_first_stage": r_ai, "human_actors": r_act}

        print(f"\n[人間の件数 log(1+n)]  n={r_asym['n_obs']:,}, repos={r_asym['n_repos']}")
        print(f"  post           = {r_asym['params']['post']:+.4f} "
              f"(se={r_asym['se_repo']['post']:.4f}, p={r_asym['p_repo']['post']:.4g})"
              f"   <- 評価工程での効果")
        print(f"  post x is_G    = {r_asym['params']['post_G']:+.4f} "
              f"(se={r_asym['se_repo']['post_G']:.4f}, p={r_asym['p_repo']['post_G']:.4g})"
              f"   <- ★生成工程は評価工程とどれだけ違うか")
        tot = r_asym["params"]["post"] + r_asym["params"]["post_G"]
        print(f"  合計(生成工程での効果) = {tot:+.4f}")
        print(f"  参考) 工程を区別しない pooled post = {r_pool['params']['post']:+.4f} "
              f"(p={r_pool['p_repo']['post']:.4g})")

        print(f"\n[人間の人数 log(1+actors)]")
        print(f"  post           = {r_act['params']['post']:+.4f} (p={r_act['p_repo']['post']:.4g})")
        print(f"  post x is_G    = {r_act['params']['post_G']:+.4f} (p={r_act['p_repo']['post_G']:.4g})")

        print(f"\n[first stage: AI の件数]")
        print(f"  post           = {r_ai['params']['post']:+.4f} (p={r_ai['p_repo']['post']:.4g})")
        print(f"  post x is_G    = {r_ai['params']['post_G']:+.4f} (p={r_ai['p_repo']['post_G']:.4g})")

        # 3) 動的効果(イベント時間 x 工程)
        k = long["month_idx"] - long["g"]
        ev_cols = []
        for j in range(-LEAD_LAG, LEAD_LAG + 1):
            if j == REF:
                continue
            nm = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
            val = (k <= j) if j == -LEAD_LAG else ((k >= j) if j == LEAD_LAG else (k == j))
            long[nm] = val.fillna(False).astype(float)
            long[nm + "_G"] = long[nm] * long["is_G"]
            ev_cols += [nm, nm + "_G"]
        r_dyn = run(long, "log_n_human", ev_cols, f"{tag} dynamic")
        out["results"][tag]["dynamic"] = r_dyn
        print(f"\n[動的効果]  {'k':>4}{'評価工程':>12}{'p':>8}{'生成-評価の差':>14}{'p':>8}")
        for j in [-3, -2, 0, 1, 2, 3]:
            nm = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
            if nm not in r_dyn["params"]:
                continue
            print(f"{'':>12}{j:>4}{r_dyn['params'][nm]:>+12.4f}{r_dyn['p_repo'][nm]:>8.3f}"
                  f"{r_dyn['params'][nm + '_G']:>+14.4f}{r_dyn['p_repo'][nm + '_G']:>8.3f}")

    (OUT_DIR / "d17_asymmetry.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd17_asymmetry.json'}")


if __name__ == "__main__":
    main()

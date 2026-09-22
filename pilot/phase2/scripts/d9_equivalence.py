"""D9: 同値性検定 — 「効果が無かった」のか「検出できなかった」のか。

D4/D7/D8 のイベントスタディは、人間側のほぼ全ての指標で効果ゼロだった
(レビュー件数、レビュアー人数、PR あたりレビュー数、承認までの日数、マージ率、
 新規参入、離脱)。しかし null result は、検出力が無いだけかもしれない。

そこで2つを報告する。
  1. 同値性検定 (TOST): 「効果が ±δ の範囲に収まっている」を積極的に示す。
     通常の検定(H0: β=0)と違い、H0: |β| >= δ を棄却することで同値性を主張する。
  2. 最小検出可能効果 (MDE): 有意水準5%・検出力80%で検出できる最小の効果の大きさ。
     |MDE| が実質的に小さければ、「意味のある効果があれば検出できたはず」と言える。

δ(等価限界)は log アウトカムで 0.10(= おおよそ10%の変化)、
比率アウトカム(H_E, merge_rate 等)で 0.05(5ポイント)を既定とする。

実行: uv run python pilot/phase2/scripts/d9_equivalence.py
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

from common import OUT_DIR

# 実質的に意味があると考える最小の効果の大きさ
DELTA_LOG = 0.10      # log アウトカム: 約10%
DELTA_RATE = 0.05     # 比率アウトカム: 5ポイント
DELTA_DAYS = 2.0      # 日数アウトカム: 2日
DELTA_COUNT = 0.5     # 件数アウトカム(レビュー往復など): 0.5件

Z_ALPHA = 1.96        # 両側5%
Z_POWER = 0.84        # 検出力80%

DELTA_BY_OUTCOME = {
    "log_n_E_human": DELTA_LOG,
    "log_n_E_agent": DELTA_LOG,
    "log_n_E_human_actors": DELTA_LOG,
    "log_n_G_human": DELTA_LOG,
    "log_ai_before_approval": DELTA_LOG,
    "log_human_before_approval": DELTA_LOG,
    "log_entrants": DELTA_LOG,
    "log_active_humans": DELTA_LOG,
    "log_entrant_pr_lines": DELTA_LOG,
    "log_entrant_pr_files": DELTA_LOG,
    "H_E": DELTA_RATE,
    "approval_rate": DELTA_RATE,
    "merge_rate": DELTA_RATE,
    "churn_rate": DELTA_RATE,
    "is_active": DELTA_RATE,
    "e_human_per_pr": DELTA_COUNT,
    "e_agent_per_pr": DELTA_COUNT,
    "avg_review_rounds": DELTA_COUNT,
    "avg_approval_lag_days": DELTA_DAYS,
    "avg_time_to_merge_days": DELTA_DAYS,
}

# 「導入が効いた後」とみなすイベント時点。k=+2, +3 を代表値に使う
POST_KEYS = ["k_p2", "k_p3"]


def tost(beta: float, se: float, delta: float) -> dict:
    """two one-sided tests。p_tost < 0.05 なら「効果は ±delta の範囲内」と言える。

    あわせて片側の non-inferiority 検定も返す。displacement 仮説は「人間が減る」という
    方向の主張なので、減少方向だけを棄却できれば十分なことが多い
    (p_no_decline < 0.05 なら「delta 以上の減少は起きていない」と言える)。
    """
    t_lower = (beta + delta) / se   # H0: beta <= -delta
    t_upper = (beta - delta) / se   # H0: beta >= +delta
    p_lower = 1 - stats.norm.cdf(t_lower)
    p_upper = stats.norm.cdf(t_upper)
    p_tost = max(p_lower, p_upper)
    return {
        "p_tost": float(p_tost),
        "equivalent_at_delta": bool(p_tost < 0.05),
        "p_no_decline": float(p_lower),          # 片側: delta 以上の減少はないか
        "no_decline_at_delta": bool(p_lower < 0.05),
        "ci95_low": float(beta - Z_ALPHA * se),
        "ci95_high": float(beta + Z_ALPHA * se),
        "mde": float((Z_ALPHA + Z_POWER) * se),
    }


def collect(path: str, section: str) -> list[dict]:
    data = json.loads((OUT_DIR / path).read_text())
    models = data.get(section, data.get("models", {}))
    rows = []
    for y, r in models.items():
        if not r or "params" not in r:
            continue
        delta = DELTA_BY_OUTCOME.get(y)
        if delta is None:
            continue
        for k in POST_KEYS:
            if k not in r["params"]:
                continue
            beta, se = r["params"][k], r["se_repo"][k]
            res = tost(beta, se, delta)
            rows.append({
                "source": path, "outcome": y, "event_time": k,
                "beta": beta, "se": se, "delta": delta,
                "p_conventional": r["p_repo"][k], **res,
            })
    return rows


def main() -> None:
    rows: list[dict] = []
    rows += collect("d4_repo_event_study.json", "models")
    rows += collect("d7_burden_event_study.json", "models")
    rows += collect("d8_talent_flow.json", "models")

    print("=== 同値性検定 (TOST) と最小検出可能効果 (MDE) ===")
    print("p_conv: 通常の検定(効果=0 の棄却)。 p_tost: 同値性の検定(|効果|>=δ の棄却)")
    print("判定: [効果あり]=p_conv<0.05 / [同値]=p_tost<0.05 / [不確定]=どちらでもない\n")
    hdr = (f"{'outcome':<26}{'k':>4}{'beta':>9}{'95%CI':>20}{'δ':>7}{'p_conv':>8}{'p_tost':>8}"
           f"{'p_nodecl':>10}{'MDE':>8}  判定")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["outcome"], x["event_time"])):
        if r["p_conventional"] < 0.05:
            verdict = "効果あり"
        elif r["equivalent_at_delta"]:
            verdict = "同値(効果は無視できる)"
        elif r["no_decline_at_delta"]:
            verdict = "減少なし(片側のみ)"
        else:
            verdict = "不確定(検出力不足)"
        ci = f"[{r['ci95_low']:+.3f},{r['ci95_high']:+.3f}]"
        print(f"{r['outcome']:<26}{r['event_time'][-2:]:>4}{r['beta']:>+9.3f}{ci:>20}"
              f"{r['delta']:>7.2f}{r['p_conventional']:>8.3f}{r['p_tost']:>8.3f}"
              f"{r['p_no_decline']:>10.4f}{r['mde']:>8.3f}  {verdict}")

    n_eff = sum(1 for r in rows if r["p_conventional"] < 0.05)
    n_equiv = sum(1 for r in rows if r["p_conventional"] >= 0.05 and r["equivalent_at_delta"])
    n_nodecl = sum(1 for r in rows if r["p_conventional"] >= 0.05
                   and not r["equivalent_at_delta"] and r["no_decline_at_delta"])
    n_incon = len(rows) - n_eff - n_equiv - n_nodecl
    print(f"\n合計 {len(rows)} 件: 効果あり {n_eff} / 同値 {n_equiv} / 減少なし(片側) {n_nodecl} / 不確定 {n_incon}")

    # 人間側の主要アウトカムについて「δ以上の減少はない」と言えるかをまとめる
    human_keys = ["log_n_E_human", "log_n_E_human_actors", "e_human_per_pr",
                  "log_active_humans", "log_entrants", "log_n_G_human"]
    print("\n=== displacement(人間の後退)仮説の検定: 「δ以上の減少はない」と言えるか ===")
    for r in sorted(rows, key=lambda x: (x["outcome"], x["event_time"])):
        if r["outcome"] in human_keys:
            mark = "言える" if r["no_decline_at_delta"] else "言えない"
            print(f"  {r['outcome']:<24} k={r['event_time'][-2:]}  "
                  f"p_no_decline={r['p_no_decline']:.4f}  -> {mark}")

    (OUT_DIR / "d9_equivalence.json").write_text(
        json.dumps({"delta": DELTA_BY_OUTCOME, "rows": rows,
                    "counts": {"effect": n_eff, "equivalent": n_equiv,
                               "no_decline_only": n_nodecl, "inconclusive": n_incon}},
                   ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\n-> {OUT_DIR / 'd9_equivalence.json'}")


if __name__ == "__main__":
    main()

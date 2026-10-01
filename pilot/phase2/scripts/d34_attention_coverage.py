"""D34: AI は人間の注意を再配分しているか — 被覆と集中度。

D33 は「疎なモジュールでは AI 導入が人間のレビューを増やす」(+0.291, p=8.4e-14)を
示した。これが本当に**再配分**なら、リポジトリ内のモジュール間で
人間の注意の集中度が下がっているはずである。

新しい目的関数(`RL_REFRAME.md` §7)を実装可能にするための指標を定義する。

  被覆     coverage(r,t) = 人間レビューが付いた PR 数 / PR 数     ← PR 単位
  集中度   gini(r,t)     = モジュール間の人間レビュー件数の Gini 係数
  効率     被覆1ポイントあたりに要した人間レビュー件数

⚠ **初版はモジュール単位で被覆を測って失敗した(2026-09-24)。**
   「活動モジュールのうち人間レビューが付いた割合」は導入前から 0.9995 で天井に張り付き、
   下がる方向にしか動けなかった。さらに活動モジュール数が 77.6 → 98.9 と 27% 増えたため、
   低下が構成変化と区別できなかった。**粒度を PR 単位に下げて測り直す。**
   集中度は構成変化を避けるため、導入前後の両方に存在するモジュールに限定する。

⚠ D15 はレビュー負荷の集中度を**人**について測っている。ここで測るのは
   **モジュール**についてであり、別の量である。

識別は D4 と同じ(リポジトリ単位の AI 導入)。単位はリポジトリ×月なので
固定効果は repo + month、クラスタは repo。

出力: outputs_full/d34_attention_coverage.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d34_attention_coverage.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import OUT_DIR, build_count_panel


def month_to_idx(s: str) -> int:
    y, m = str(s).split("-")[:2]
    return int(y) * 12 + int(m)


def gini(x: np.ndarray) -> float:
    """Gini 係数。全員ゼロなら 0 を返す。"""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1).dot(x) / (n * x.sum()))


def top_share(x: np.ndarray, q: float = 0.1) -> float:
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    k = max(1, int(np.ceil(len(x) * q)))
    return float(x[:k].sum() / x.sum())


def main() -> None:
    adoption = json.loads((OUT_DIR / "d4_repo_event_study.json").read_text())["adoption"]
    # ★対照は never-treated のみ。判定不能で除外したリポジトリを対照に混ぜない(2026-09-27 数値表の指摘)
    _d4 = json.loads((OUT_DIR / "d4_repo_event_study.json").read_text())
    KEEP_REPOS = set(adoption) | set(_d4["never_treated"])

    # --- PR 単位の被覆 ---
    d5 = pd.read_parquet(OUT_DIR / "d5_pr_level.parquet")
    d5 = d5[d5["repo"].isin(KEEP_REPOS)]
    d5["covered"] = (d5["n_human_reviews_total"] > 0).astype(float)
    pr = d5.groupby(["repo", "month"]).agg(
        n_prs=("covered", "size"),
        coverage=("covered", "mean"),
        n_E_human=("n_human_reviews_total", "sum"),
        n_E_agent=("n_ai_reviews_total", "sum"),
    ).reset_index()

    # --- モジュール間の集中度(構成変化を避けるため均衡パネルに限定)---
    df = build_count_panel()
    df = df[df["repo"].isin(KEEP_REPOS)].copy()
    act = df[df["n_E_prs"] > 0].copy()
    act["t"] = act["month"].map(month_to_idx)
    act["adopt_t"] = act["repo"].map(
        lambda r: month_to_idx(adoption[r]) if r in adoption else np.nan)
    act["post"] = (act["adopt_t"].notna() & (act["t"] >= act["adopt_t"])).astype(float)
    # 処置リポジトリでは導入前後の両方に、対照では前半後半の両方に出現するモジュールだけ残す
    med = act.groupby("repo")["t"].transform("median")
    act["half"] = np.where(act["adopt_t"].notna(), act["post"], (act["t"] >= med).astype(float))
    both = act.groupby(["module_id_global"])["half"].nunique()
    keep = set(both[both == 2].index)
    bal = act[act["module_id_global"].isin(keep)]
    print(f"均衡モジュール {len(keep):,} / 全 {act['module_id_global'].nunique():,}")

    rows = []
    for (repo, month), g in bal.groupby(["repo", "month"]):
        h = g["n_E_human"].to_numpy()
        rows.append({"repo": repo, "month": month, "n_modules": len(g),
                     "gini_modules": gini(h), "top10_modules": top_share(h, 0.1)})
    conc = pd.DataFrame(rows)

    rt = pr.merge(conc, on=["repo", "month"], how="inner")
    rt["t"] = rt["month"].map(month_to_idx)
    rt["adopt_t"] = rt["repo"].map(lambda r: month_to_idx(adoption[r]) if r in adoption else np.nan)
    rt["post"] = (rt["adopt_t"].notna() & (rt["t"] >= rt["adopt_t"])).astype(float)
    rt["log_n_E_human"] = np.log1p(rt["n_E_human"])
    # 効率: 被覆した PR 1件あたりに要した人間レビュー件数(小さいほど効率的)
    rt["reviews_per_covered"] = rt["n_E_human"] / np.maximum(rt["coverage"] * rt["n_prs"], 1)

    print(f"リポジトリ×月 {len(rt):,} 行 / リポジトリ {rt['repo'].nunique()}")
    print(f"処置 {rt[rt['adopt_t'].notna()]['repo'].nunique()} / "
          f"対照 {rt[rt['adopt_t'].isna()]['repo'].nunique()}\n")

    print("=== 記述: 導入前後の平均(処置リポジトリのみ)===")
    tre = rt[rt["adopt_t"].notna()]
    print(tre.groupby("post")[["coverage", "gini_modules", "top10_modules",
                               "n_E_human", "n_prs", "reviews_per_covered"]].mean()
          .to_string(float_format=lambda x: f"{x:10.4f}"))

    import statsmodels.formula.api as smf
    out: dict = {"n_obs": int(len(rt)), "n_balanced_modules": len(keep), "models": {}}
    print("\n=== DiD: repo + month 固定効果、repo クラスタ ===")
    print(f"{'被説明変数':26s} {'係数':>10s} {'se':>9s} {'p':>11s}")
    for y, lab in [("coverage", "被覆率(PR単位)"),
                   ("gini_modules", "集中度(Gini・均衡)"),
                   ("top10_modules", "上位10%シェア(均衡)"),
                   ("log_n_E_human", "log 人間レビュー数"),
                   ("reviews_per_covered", "被覆PR1件あたりレビュー")]:
        sub = rt.dropna(subset=[y])
        m = smf.ols(f"{y} ~ post + C(repo) + C(month)", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["repo"]})
        c, se, p = m.params["post"], m.bse["post"], m.pvalues["post"]
        star = " *" if p < 0.05 else ""
        print(f"  {lab:26s} {c:+10.4f} {se:9.4f} {p:11.4g}{star}")
        out["models"][y] = {"coef": float(c), "se": float(se), "p": float(p),
                            "n_obs": int(m.nobs)}

    print("\n  予測: 再配分が起きているなら 被覆率↑ / 集中度↓")
    (OUT_DIR / "d34_attention_coverage.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n-> {OUT_DIR / 'd34_attention_coverage.json'}")


if __name__ == "__main__":
    main()

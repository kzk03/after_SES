"""D15: レビュー負荷は誰に乗っているのか — AI 導入で分配は変わるか。

ここまでで「AI が入っても人間のレビュー総量は変わらない」ことが分かった。
しかし総量が同じでも、**誰が担っているか**は変わりうる。
OSS の持続可能性の文脈では、少数のコアメンテナへの負荷集中が問題とされてきた
(元の SES 研究「レビュー継続予測」が扱っていたのもこの問題系)。

そこで2つを見る。
  A. 集中度: モジュール x 月の人間レビュー件数の Gini 係数・上位1人シェア
  B. 層別: コア(リポジトリ内レビュー件数上位10%)と周辺で、レビュー量の変化が違うか

「AI が周辺貢献者の軽いレビューを肩代わりし、コアの負担だけが残る」なら集中が進む。
「AI がコアの負担を肩代わりする」なら集中は緩む。どちらでもないなら分配も不変。

実行: uv run python pilot/phase2/scripts/d15_burden_distribution.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from common import INTERIM_DIR, OUT_DIR, build_count_panel, load_repos, repo_dir_name
from d4_repo_event_study import LEAD_LAG, REF, detect_adoption

CORE_PCTL = 0.90  # リポジトリ内レビュー件数のこの分位以上をコアとみなす


def gini(x: np.ndarray) -> float:
    if len(x) < 2 or x.sum() == 0:
        return np.nan
    xs = np.sort(x)
    n = len(xs)
    cum = np.cumsum(xs)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def build_reviewer_panel() -> pd.DataFrame:
    """module x month x (コア/周辺) の人間レビュー件数と、集中度指標を作る。"""
    rows = []
    conc_rows = []
    for repo in load_repos():
        ev_path = INTERIM_DIR / repo_dir_name(repo) / "events.parquet"
        if not ev_path.exists():
            continue
        ev = pd.read_parquet(ev_path)
        e = ev[(ev["stage"] == "E") & (ev["actor_category"] == "human")]
        if e.empty:
            continue
        # コアの定義: そのリポジトリでの期間全体のレビュー件数が上位10%
        per_actor = e.groupby("actor_login").size()
        thr = per_actor.quantile(CORE_PCTL)
        core = set(per_actor[per_actor >= thr].index)

        e = e.copy()
        e["is_core"] = e["actor_login"].isin(core)
        g = e.groupby(["module_id", "month", "is_core"]).size().rename("n").reset_index()
        g["repo"] = repo
        rows.append(g)

        # 集中度: module x month で各レビュアーの件数分布
        for (mod, mon), sub in e.groupby(["module_id", "month"]):
            cnt = sub.groupby("actor_login").size().values
            if len(cnt) < 2:
                continue
            conc_rows.append({
                "repo": repo,
                "module_id_global": f"{repo}::{mod}",
                "month": mon,
                "n_reviewers": len(cnt),
                "gini": gini(cnt),
                "top1_share": float(cnt.max() / cnt.sum()),
                "hhi": float(((cnt / cnt.sum()) ** 2).sum()),
            })

    layer = pd.concat(rows, ignore_index=True)
    layer["module_id_global"] = layer["repo"] + "::" + layer["module_id"]
    conc = pd.DataFrame(conc_rows)
    return layer, conc


def add_event(d: pd.DataFrame, adoption: dict, midx: dict, never: list) -> tuple[pd.DataFrame, list[str]]:
    d = d[d["repo"].isin(set(adoption) | set(never))].copy()
    d["month_period"] = pd.PeriodIndex(d["month"], freq="M").to_timestamp()
    d["month_idx"] = d["month"].map(midx)
    d["g"] = d["repo"].map({r: midx[m] for r, m in adoption.items()})
    k = d["month_idx"] - d["g"]
    cols = []
    for j in range(-LEAD_LAG, LEAD_LAG + 1):
        if j == REF:
            continue
        name = f"k_m{abs(j)}" if j < 0 else f"k_p{j}"
        val = (k <= j) if j == -LEAD_LAG else ((k >= j) if j == LEAD_LAG else (k == j))
        d[name] = val.fillna(False).astype(float)
        cols.append(name)
    return d, cols


def fit(d: pd.DataFrame, y: str, cols: list[str]) -> dict:
    dd = d.dropna(subset=[y, *cols]).set_index(["module_id_global", "month_period"])
    r = PanelOLS(dd[y], dd[cols], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", clusters=dd["repo"]
    )
    return {"params": r.params.to_dict(), "p_repo": r.pvalues.to_dict(),
            "se_repo": r.std_errors.to_dict(), "n_obs": int(r.nobs)}


def show(tag: str, r: dict) -> None:
    b2, p2 = r["params"].get("k_p2", np.nan), r["p_repo"].get("k_p2", np.nan)
    b3, p3 = r["params"].get("k_p3", np.nan), r["p_repo"].get("k_p3", np.nan)
    m2, pm2 = r["params"].get("k_m2", np.nan), r["p_repo"].get("k_m2", np.nan)
    print(f"  {tag:<28}{m2:>+10.4f}{pm2:>8.3f}{b2:>+10.4f}{p2:>8.3f}{b3:>+10.4f}{p3:>8.3f}"
          f"{r['n_obs']:>9,}")


def main() -> None:
    df = build_count_panel()
    months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(months)}
    adoption, never, always = detect_adoption(df)

    layer, conc = build_reviewer_panel()
    out: dict = {"core_pctl": CORE_PCTL, "models": {}}

    # --- 記述 --------------------------------------------------------------------
    tot = layer.groupby("is_core")["n"].sum()
    print("=== 人間のレビュー件数に占めるコア/周辺の割合(全期間) ===")
    print(f"  コア(上位10%のレビュアー): {tot[True]:>9,} 件 ({tot[True]/tot.sum():.1%})")
    print(f"  周辺                      : {tot[False]:>9,} 件 ({tot[False]/tot.sum():.1%})")
    out["core_share_of_reviews"] = float(tot[True] / tot.sum())

    print(f"\n  集中度の平均: Gini={conc['gini'].mean():.3f}  "
          f"上位1人シェア={conc['top1_share'].mean():.3f}  "
          f"(レビュアー2人以上の module x month, n={len(conc):,})")

    # --- A. 集中度のイベントスタディ -----------------------------------------------
    conc_e, cols = add_event(conc, adoption, midx, never)
    print("\n=== A. レビュー負荷の集中度(処置=AI レビュー導入) ===")
    print(f"  {'被説明変数':<28}{'k=-2':>10}{'p':>8}{'k=+2':>10}{'p':>8}{'k=+3':>10}{'p':>8}{'n':>9}")
    for y in ["gini", "top1_share", "hhi", "n_reviewers"]:
        r = fit(conc_e, y, cols)
        out["models"][f"conc_{y}"] = r
        show(y, r)

    # --- B. コア/周辺 別のレビュー量 ------------------------------------------------
    # module x month x is_core を module x month の2列に展開してから推定する
    wide = layer.pivot_table(index=["repo", "module_id_global", "month"], columns="is_core",
                             values="n", aggfunc="sum").fillna(0.0)
    wide.columns = ["n_peripheral", "n_core"]
    wide = wide.reset_index()
    wide["log_n_core"] = np.log1p(wide["n_core"])
    wide["log_n_peripheral"] = np.log1p(wide["n_peripheral"])
    wide["core_share"] = np.where(
        wide["n_core"] + wide["n_peripheral"] > 0,
        wide["n_core"] / (wide["n_core"] + wide["n_peripheral"]), np.nan)
    wide_e, cols2 = add_event(wide, adoption, midx, never)

    print("\n=== B. コア / 周辺 別のレビュー量 ===")
    print(f"  {'被説明変数':<28}{'k=-2':>10}{'p':>8}{'k=+2':>10}{'p':>8}{'k=+3':>10}{'p':>8}{'n':>9}")
    for y in ["log_n_core", "log_n_peripheral", "core_share"]:
        r = fit(wide_e, y, cols2)
        out["models"][y] = r
        show(y, r)

    (OUT_DIR / "d15_burden_distribution.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n-> {OUT_DIR / 'd15_burden_distribution.json'}")


if __name__ == "__main__":
    main()

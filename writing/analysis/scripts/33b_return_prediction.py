"""33b: 復帰予測 (0->1) の検証。

論文動機「貢献再開」に対応。区間間で 8.4% のペアが復帰(前区間 label=0 -> 次区間 label=1)する。
離脱したレビュアーの復帰を IRL/RF/Ensemble で予測できるかを検証する。

タスク定義(3連続遷移 + pooled):
  - 復帰予測 (return)   : 前区間 label=0 のペアを母集団, 次区間 label=1 を正例
  - 継続予測 (continue) : 前区間 label=1 のペアを母集団, 次区間 label=1 を正例
  - 離脱予測 (drop)     : 前区間 label=1 のペアを母集団, 次区間 label=0 を正例(=非継続)

スコア源(CLAUDE.md 規約の3種のみ):
  - cold IRL : irl_{h}
  - RF       : rf_{h}
  - Ensemble : avg_{h} = (irl+rf)/2  ※ 既存列を使用

評価は rank-based (AUC) のみ。cold IRL は未校正なので閾値・絶対値は使わない。
bootstrap CI (seed=777) を AUC に付与。
出力: results/*.csv, results/fig_return_prediction.pdf
"""

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 777
BASE = "/Users/kazuki-h/Master_research/writing/analysis"
DATA = f"{BASE}/data"
RESULTS = f"{BASE}/results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
TRANSITIONS = [("0-3m", "3-6m"), ("3-6m", "6-9m"), ("6-9m", "9-12m")]
MODELS = {"IRL": "irl", "RF": "rf", "Ensemble": "avg"}
N_BOOT = 2000


def boot_auc_ci(y, s, n_boot=N_BOOT, seed=RANDOM_SEED):
    """AUC の bootstrap 95%CI。両クラス揃う場合のみ。"""
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y, s)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
    if not aucs:
        return point, np.nan, np.nan
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return point, lo, hi


def build_task_rows(df, task):
    """各遷移について (prev_h, next_h, score列, y) を返す。

    task: 'return' / 'continue' / 'drop'
    return  : prev==0 母集団, 正例= next==1
    continue: prev==1 母集団, 正例= next==1
    drop    : prev==1 母集団, 正例= next==0
    各 score は「次区間で label=1 になる確率」を予測する向き。
    drop は「非継続(=label=0)」が正例なので score を反転 (-prob) して使う。
    """
    out = {}
    for prev_h, next_h in TRANSITIONS:
        if task == "return":
            sub = df[df[f"label_{prev_h}"] == 0].copy()
            y = (sub[f"label_{next_h}"] == 1).astype(int).values
            flip = False
        elif task == "continue":
            sub = df[df[f"label_{prev_h}"] == 1].copy()
            y = (sub[f"label_{next_h}"] == 1).astype(int).values
            flip = False
        elif task == "drop":
            sub = df[df[f"label_{prev_h}"] == 1].copy()
            y = (sub[f"label_{next_h}"] == 0).astype(int).values
            flip = True
        else:
            raise ValueError(task)
        out[(prev_h, next_h)] = (sub, y, flip)
    return out


def main():
    df = pd.read_csv(f"{DATA}/pair_trajectories.csv")
    df["top_dir"] = df["directory"].str.split("/").str[0]

    # ============================================================
    # 1) 復帰タスクの base rate と AUC (3遷移 + pooled)
    # ============================================================
    rows = []
    pooled = {task: {m: {"y": [], "score": []} for m in MODELS} for task in
              ["return", "continue", "drop"]}
    pooled_base = {task: {"n": 0, "pos": 0} for task in ["return", "continue", "drop"]}

    for task in ["return", "continue", "drop"]:
        tdata = build_task_rows(df, task)
        for (prev_h, next_h), (sub, y, flip) in tdata.items():
            n = len(sub)
            pos = int(y.sum())
            base = pos / n if n else np.nan
            pooled_base[task]["n"] += n
            pooled_base[task]["pos"] += pos
            for mname, mcol in MODELS.items():
                # score: 「次区間 label=1 確率」。drop は反転して「非継続」確率に。
                prob = sub[f"{mcol}_{next_h}" if mcol != "avg" else f"avg_{next_h}"].values
                score = -prob if flip else prob
                auc, lo, hi = boot_auc_ci(y, score)
                rows.append({
                    "task": task, "transition": f"{prev_h}->{next_h}",
                    "next_h": next_h, "model": mname,
                    "n": n, "pos": pos, "base_rate": base,
                    "auc": auc, "auc_lo": lo, "auc_hi": hi,
                })
                pooled[task][mname]["y"].append(y)
                pooled[task][mname]["score"].append(score)

    # pooled (全遷移を結合)
    for task in ["return", "continue", "drop"]:
        n = pooled_base[task]["n"]
        pos = pooled_base[task]["pos"]
        base = pos / n if n else np.nan
        for mname in MODELS:
            y = np.concatenate(pooled[task][mname]["y"])
            score = np.concatenate(pooled[task][mname]["score"])
            auc, lo, hi = boot_auc_ci(y, score)
            rows.append({
                "task": task, "transition": "pooled", "next_h": "pooled",
                "model": mname, "n": n, "pos": pos, "base_rate": base,
                "auc": auc, "auc_lo": lo, "auc_hi": hi,
            })

    res = pd.DataFrame(rows)
    res.to_csv(f"{RESULTS}/return_prediction_auc.csv", index=False)

    # ============================================================
    # 3) 復帰当て成否の特徴量差 (pooled return タスク)
    #    feature_values t0='2022-01-01' を developer 単位で結合。
    #    「復帰した(y=1)ペア」のうち、IRL/Ens が上位に当てられたか。
    # ============================================================
    feat = pd.read_csv(f"{DATA}/feature_values.csv")
    feat = feat[feat["__snapshot"] == "2022-01-01"].copy()
    feat = feat.rename(columns={"__email": "developer"})
    feat_cols = [
        "recent_activity_frequency", "avg_activity_gap", "days_since_last_activity",
        "overall_acceptance_rate", "recent_acceptance_rate", "active_months_ratio",
        "total_reviews", "core_reviewer_ratio", "activity_trend",
    ]
    feat_dev = feat.groupby("developer")[feat_cols].mean().reset_index()

    # repo 経験プロキシも結合 (pair 単位)
    repoexp = pd.read_csv(f"{DATA}/pair_repo_experience.csv")

    # pooled return 母集団を再構築し、各ペアの予測スコアと正解を保持
    ret_records = []
    for prev_h, next_h in TRANSITIONS:
        sub = df[df[f"label_{prev_h}"] == 0].copy()
        sub = sub.assign(
            transition=f"{prev_h}->{next_h}",
            returned=(sub[f"label_{next_h}"] == 1).astype(int),
            irl_score=sub[f"irl_{next_h}"],
            rf_score=sub[f"rf_{next_h}"],
            ens_score=sub[f"avg_{next_h}"],
        )
        ret_records.append(sub[["developer", "directory", "top_dir", "transition",
                                 "returned", "irl_score", "rf_score", "ens_score"]])
    ret_df = pd.concat(ret_records, ignore_index=True)

    # 「予測成否」: 各遷移内でスコア上位50%を「復帰すると予測」とみなす(rank-based, 校正不要)
    for col, lab in [("irl_score", "irl"), ("rf_score", "rf"), ("ens_score", "ens")]:
        ret_df[f"{lab}_pred_pos"] = (
            ret_df.groupby("transition")[col].transform(
                lambda x: x >= x.median()).astype(int)
        )
    # IRL が当てた復帰 = returned==1 & irl_pred_pos==1 (true positive)
    ret_df["irl_hit"] = ((ret_df["returned"] == 1) & (ret_df["irl_pred_pos"] == 1)).astype(int)
    ret_df["irl_miss"] = ((ret_df["returned"] == 1) & (ret_df["irl_pred_pos"] == 0)).astype(int)

    ret_feat = ret_df.merge(feat_dev, on="developer", how="left")
    ret_feat = ret_feat.merge(
        repoexp[["developer", "directory", "prior_n", "prior_acc",
                 "log_prior_n", "prior_recency_days"]],
        on=["developer", "directory"], how="left",
    )
    ret_feat.to_csv(f"{RESULTS}/return_pair_features.csv", index=False)

    # 復帰ペアのうち IRL が当てた(hit) vs 外した(miss) の特徴量差
    returned = ret_feat[ret_feat["returned"] == 1].copy()
    cmp_cols = feat_cols + ["prior_n", "prior_acc", "log_prior_n", "prior_recency_days"]
    diff_rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for c in cmp_cols:
        hit_v = returned.loc[returned["irl_hit"] == 1, c].dropna()
        miss_v = returned.loc[returned["irl_miss"] == 1, c].dropna()
        if len(hit_v) < 5 or len(miss_v) < 5:
            continue
        # 標準化差(Cohen's d 風) + bootstrap CI of mean diff
        pooled_sd = np.sqrt((hit_v.var(ddof=1) + miss_v.var(ddof=1)) / 2)
        d = (hit_v.mean() - miss_v.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        # bootstrap mean diff
        diffs = []
        hv, mv = hit_v.values, miss_v.values
        for _ in range(N_BOOT):
            db = rng.choice(hv, len(hv)).mean() - rng.choice(mv, len(mv)).mean()
            diffs.append(db)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        diff_rows.append({
            "feature": c,
            "hit_mean": hit_v.mean(), "miss_mean": miss_v.mean(),
            "mean_diff": hit_v.mean() - miss_v.mean(),
            "diff_lo": lo, "diff_hi": hi,
            "cohens_d": d, "n_hit": len(hit_v), "n_miss": len(miss_v),
        })
    diff_df = pd.DataFrame(diff_rows).sort_values("cohens_d", key=abs, ascending=False)
    diff_df.to_csv(f"{RESULTS}/return_hit_feature_diff.csv", index=False)

    # 復帰した vs しなかった (0群内) の特徴量差 — 「復帰しやすいペア」記述
    desc_rows = []
    for c in cmp_cols:
        r1 = ret_feat.loc[ret_feat["returned"] == 1, c].dropna()
        r0 = ret_feat.loc[ret_feat["returned"] == 0, c].dropna()
        if len(r1) < 5 or len(r0) < 5:
            continue
        pooled_sd = np.sqrt((r1.var(ddof=1) + r0.var(ddof=1)) / 2)
        d = (r1.mean() - r0.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        desc_rows.append({
            "feature": c, "returned_mean": r1.mean(), "stayed0_mean": r0.mean(),
            "mean_diff": r1.mean() - r0.mean(), "cohens_d": d,
            "n_returned": len(r1), "n_stayed0": len(r0),
        })
    desc_df = pd.DataFrame(desc_rows).sort_values("cohens_d", key=abs, ascending=False)
    desc_df.to_csv(f"{RESULTS}/return_propensity_feature_diff.csv", index=False)

    # ============================================================
    # 図 fig_return_prediction.pdf (3パネル)
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) 3タスク × 3モデルの pooled AUC 棒
    ax = axes[0]
    task_labels = {"continue": "継続予測\n(前1→次1)", "drop": "離脱予測\n(前1→次0)",
                   "return": "復帰予測\n(前0→次1)"}
    task_order = ["continue", "drop", "return"]
    model_order = ["IRL", "RF", "Ensemble"]
    colors = {"IRL": "#d62728", "RF": "#1f77b4", "Ensemble": "#2ca02c"}
    x = np.arange(len(task_order))
    w = 0.25
    pooled_res = res[res["transition"] == "pooled"]
    for i, m in enumerate(model_order):
        vals, los, his = [], [], []
        for t in task_order:
            row = pooled_res[(pooled_res["task"] == t) & (pooled_res["model"] == m)].iloc[0]
            vals.append(row["auc"])
            los.append(row["auc"] - row["auc_lo"])
            his.append(row["auc_hi"] - row["auc"])
        ax.bar(x + (i - 1) * w, vals, w, label=m, color=colors[m],
               yerr=[los, his], capsize=3, error_kw={"elinewidth": 1})
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([task_labels[t] for t in task_order])
    ax.set_ylabel("AUC (pooled, 95%CI)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("(a) 3タスク × 3モデルの予測性能")
    ax.legend(loc="lower left", fontsize=9)

    # (b) 復帰予測の区間別 AUC
    ax = axes[1]
    ret_res = res[(res["task"] == "return") & (res["transition"] != "pooled")]
    trans_order = [f"{a}->{b}" for a, b in TRANSITIONS]
    x2 = np.arange(len(trans_order))
    for i, m in enumerate(model_order):
        vals, los, his = [], [], []
        for tr in trans_order:
            row = ret_res[(ret_res["transition"] == tr) & (ret_res["model"] == m)].iloc[0]
            vals.append(row["auc"])
            los.append(row["auc"] - row["auc_lo"])
            his.append(row["auc_hi"] - row["auc"])
        ax.bar(x2 + (i - 1) * w, vals, w, label=m, color=colors[m],
               yerr=[los, his], capsize=3, error_kw={"elinewidth": 1})
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(x2)
    ax.set_xticklabels(trans_order, fontsize=9)
    ax.set_ylabel("AUC (95%CI)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("(b) 復帰予測の区間別 AUC")
    ax.legend(loc="lower left", fontsize=9)

    # (c) 復帰当て成否(IRL hit vs miss)の特徴量差 (Cohen's d top)
    ax = axes[2]
    top = diff_df.head(7).iloc[::-1]
    feat_jp = {
        "recent_activity_frequency": "直近活動頻度",
        "avg_activity_gap": "平均活動間隔",
        "days_since_last_activity": "最終活動からの日数",
        "overall_acceptance_rate": "通算承諾率",
        "recent_acceptance_rate": "直近承諾率",
        "active_months_ratio": "活動月比率",
        "total_reviews": "通算レビュー数",
        "core_reviewer_ratio": "コア比率",
        "activity_trend": "活動トレンド",
        "prior_n": "事前repo経験数",
        "prior_acc": "事前repo承諾率",
        "log_prior_n": "log事前repo経験",
        "prior_recency_days": "事前repo直近日数",
    }
    yy = np.arange(len(top))
    bcol = ["#d62728" if v > 0 else "#1f77b4" for v in top["cohens_d"]]
    ax.barh(yy, top["cohens_d"], color=bcol)
    ax.set_yticks(yy)
    ax.set_yticklabels([feat_jp.get(f, f) for f in top["feature"]], fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Cohen's d (IRL hit − miss, 復帰ペア内)")
    ax.set_title("(c) IRL が復帰を当てた/外したペアの特徴量差")

    plt.tight_layout()
    fig.savefig(f"{RESULTS}/fig_return_prediction.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # コンソール要約
    # ============================================================
    print("\n===== タスク別 base rate (pooled) =====")
    for t in task_order:
        row = pooled_res[(pooled_res["task"] == t) & (pooled_res["model"] == "IRL")].iloc[0]
        print(f"  {t:9s}: n={row['n']:5d} pos={row['pos']:5d} base_rate={row['base_rate']:.4f}")

    print("\n===== pooled AUC (タスク × モデル) =====")
    for t in task_order:
        line = f"  {t:9s}: "
        for m in model_order:
            row = pooled_res[(pooled_res["task"] == t) & (pooled_res["model"] == m)].iloc[0]
            line += f"{m}={row['auc']:.3f}[{row['auc_lo']:.3f},{row['auc_hi']:.3f}]  "
        print(line)

    print("\n===== 復帰予測: 区間別 AUC =====")
    for tr in trans_order:
        line = f"  {tr}: "
        for m in model_order:
            row = ret_res[(ret_res["transition"] == tr) & (ret_res["model"] == m)].iloc[0]
            line += f"{m}={row['auc']:.3f}  "
        print(line)

    print("\n===== 復帰しやすいペア(returned vs stayed0) top特徴 =====")
    print(desc_df.head(6).to_string(index=False))

    print("\n===== IRL が復帰を当てた(hit) vs 外した(miss) top特徴 =====")
    print(diff_df.head(6).to_string(index=False))

    # 最も予測しやすいモデルの特定 (復帰 pooled)
    ret_pooled = pooled_res[pooled_res["task"] == "return"].sort_values("auc", ascending=False)
    best = ret_pooled.iloc[0]
    print(f"\n>>> 復帰予測の最良モデル: {best['model']} "
          f"AUC={best['auc']:.3f} [{best['auc_lo']:.3f},{best['auc_hi']:.3f}]")
    cont_best = pooled_res[(pooled_res["task"] == "continue")].sort_values(
        "auc", ascending=False).iloc[0]
    drop_best = pooled_res[(pooled_res["task"] == "drop")].sort_values(
        "auc", ascending=False).iloc[0]
    print(f">>> 継続予測 最良: {cont_best['model']} AUC={cont_best['auc']:.3f}")
    print(f">>> 離脱予測 最良: {drop_best['model']} AUC={drop_best['auc']:.3f}")


if __name__ == "__main__":
    main()

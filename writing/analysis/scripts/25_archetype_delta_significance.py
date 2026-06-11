# -*- coding: utf-8 -*-
"""archetype 補完性 (Δ=IRL−RF AUC) の統計的裏付け [W1 rigor].

論文 §5.2 (RQ2) の Table tab:dev-arch / tab:pair-arch-delta は archetype × 予測区間の
Δ AUC を**点推定のみ**で示しており、CI も有意性検定もない。本スクリプトは次の 2 つを足す:

  (1) 各 (archetype, 予測区間) セルの Δ=IRL_AUC−RF_AUC に **開発者クラスタ bootstrap 95% CI**
      を付与する。同一開発者が複数ペアを持つ開発者内相関を尊重するため、ペア単位ではなく
      **開発者単位で再標本化** (論文 §4.3 の懸念に直接対応)。CI が 0 を跨がないセルを「有意」とする。

  (2) 「model の優劣が archetype に依存する (= 補完性は偶然でない)」ことの **並べ替え検定**。
      archetype ラベルを (開発者単位 / ペア単位で) シャッフルし、archetype 間の Δ のばらつき
      (max−min レンジの 4 予測区間平均) が観測値以上になる確率を p 値とする。

cold IRL は RULES 準拠で未校正のため、評価はすべて rank-based (AUC) で行う。

入力:
  data/pair_trajectories.csv   (archetype = 開発者 archetype A1-A4)
  data/pair_archetype.csv      (dominant  = ペア archetype A1-A4 → P1-P4)

出力:
  results/archetype_delta_ci.csv          32 行 (2 粒度 × 4 archetype × 4 予測区間)
  results/archetype_delta_permutation.json 並べ替え検定 (粒度別)
  results/fig_archetype_delta_ci.pdf      Δ の forest plot (CI 付き)
"""
from __future__ import annotations

import json
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
HORIZON_LABEL_JP = {"0-3m": "0-3 か月", "3-6m": "3-6 か月",
                    "6-9m": "6-9 か月", "9-12m": "9-12 か月"}
CODES = ["A1", "A2", "A3", "A4"]
PMAP = {"A1": "P1", "A2": "P2", "A3": "P3", "A4": "P4"}

DEV_NAME = {"A1": "A1 低活動の新人", "A2": "A2 レビュー専念型",
            "A3": "A3 高承諾のベテラン", "A4": "A4 常時活動型"}
PAIR_NAME = {"P1": "P1 ディレクトリ未経験", "P2": "P2 経験あり低承諾",
             "P3": "P3 経験あり高承諾", "P4": "P4 中程度関与"}

N_BOOT = 2000
N_PERM = 2000
SEED = 777


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def cell_delta(d: pd.DataFrame, h: str) -> tuple[float, float, float]:
    """(archetype, horizon) セルの pooled IRL_AUC, RF_AUC, Δ を返す."""
    y = d[f"label_{h}"].astype(int).values
    irl_auc = safe_auc(y, d[f"irl_{h}"].values)
    rf_auc = safe_auc(y, d[f"rf_{h}"].values)
    return irl_auc, rf_auc, irl_auc - rf_auc


def cluster_bootstrap_ci(d: pd.DataFrame, h: str, rng: np.random.Generator,
                         n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """開発者クラスタ bootstrap で Δ の 95% CI と two-sided 経験 p 値を返す.

    開発者を復元抽出 → その開発者の全ペア行を集めて pooled Δ を再計算。
    両クラスが揃わない再標本は欠損として除外する。
    """
    devs = d["developer"].unique()
    # 開発者 → 行インデックス配列の辞書 (高速化)
    idx_by_dev = {dev: d.index[d["developer"] == dev].to_numpy() for dev in devs}
    y_all = d[f"label_{h}"].astype(int)
    irl_all = d[f"irl_{h}"]
    rf_all = d[f"rf_{h}"]

    deltas = np.empty(n_boot)
    deltas[:] = np.nan
    n_dev = len(devs)
    for b in range(n_boot):
        pick = rng.integers(0, n_dev, size=n_dev)
        rows = np.concatenate([idx_by_dev[devs[i]] for i in pick])
        y = y_all.loc[rows].values
        if len(np.unique(y)) < 2:
            continue
        di = roc_auc_score(y, irl_all.loc[rows].values)
        dr = roc_auc_score(y, rf_all.loc[rows].values)
        deltas[b] = di - dr
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) < 50:
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided 経験 p 値: 0 が分布のどちら側の裾にあるか
    p_gt = np.mean(deltas <= 0)
    p_lt = np.mean(deltas >= 0)
    p = float(min(1.0, 2 * min(p_gt, p_lt)))
    return float(lo), float(hi), p


def compute_ci_table(df: pd.DataFrame, gran: str, arch_col: str,
                     codes: list[str], name_map: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for code in codes:
        sub_arch = df[df[arch_col] == code]
        for h in HORIZONS:
            d = sub_arch.dropna(subset=[f"label_{h}", f"irl_{h}", f"rf_{h}"]).copy()
            d = d.reset_index(drop=True)
            n = len(d)
            n_pos = int(d[f"label_{h}"].sum())
            irl_auc, rf_auc, delta = cell_delta(d, h)
            lo, hi, p = cluster_bootstrap_ci(d, h, rng)
            sig = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
            rows.append({
                "granularity": gran,
                "archetype": name_map.get(code, code).split()[0]
                if gran == "pair" else code,
                "archetype_name": name_map.get(code, code),
                "horizon": h,
                "n": n, "n_pos": n_pos,
                "base_rate": round(n_pos / n, 4) if n else np.nan,
                "irl_auc": round(irl_auc, 4), "rf_auc": round(rf_auc, 4),
                "delta_pp": round(delta * 100, 2),
                "ci_lo_pp": round(lo * 100, 2), "ci_hi_pp": round(hi * 100, 2),
                "boot_p": round(p, 4), "sig_ci_excl_0": sig,
            })
    return pd.DataFrame(rows)


def horizon_averaged_delta_ci(df: pd.DataFrame, arch_col: str, code: str,
                              rng: np.random.Generator,
                              n_boot: int = N_BOOT) -> dict:
    """1 archetype の「4 予測区間平均 Δ」に開発者クラスタ bootstrap CI を付与.

    各 bootstrap 再標本 (開発者復元抽出) で 4 予測区間それぞれの Δ を計算し平均する。
    A2 のように個別区間では有意でなくとも符号が一貫するケースを 1 つの統計量で評価する。
    """
    sub = df[df[arch_col] == code].copy().reset_index(drop=True)
    devs = sub["developer"].unique()
    idx_by_dev = {d: sub.index[sub["developer"] == d].to_numpy() for d in devs}

    # 観測値: 4 予測区間 Δ の平均
    obs = []
    for h in HORIZONS:
        d = sub.dropna(subset=[f"label_{h}", f"irl_{h}", f"rf_{h}"])
        _, _, delta = cell_delta(d, h)
        obs.append(delta)
    obs_mean = float(np.nanmean(obs))

    means = np.full(n_boot, np.nan)
    n_dev = len(devs)
    for b in range(n_boot):
        pick = rng.integers(0, n_dev, size=n_dev)
        rows = np.concatenate([idx_by_dev[devs[i]] for i in pick])
        bs = sub.loc[rows]
        ds = []
        for h in HORIZONS:
            d = bs.dropna(subset=[f"label_{h}", f"irl_{h}", f"rf_{h}"])
            y = d[f"label_{h}"].astype(int).values
            if len(np.unique(y)) < 2:
                continue
            ds.append(roc_auc_score(y, d[f"irl_{h}"].values)
                      - roc_auc_score(y, d[f"rf_{h}"].values))
        if ds:
            means[b] = np.mean(ds)
    means = means[~np.isnan(means)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_gt = np.mean(means <= 0)
    p_lt = np.mean(means >= 0)
    p = float(min(1.0, 2 * min(p_gt, p_lt)))
    return {
        "delta_pp": round(obs_mean * 100, 2),
        "ci_lo_pp": round(float(lo) * 100, 2),
        "ci_hi_pp": round(float(hi) * 100, 2),
        "boot_p": round(p, 4),
        "sig": bool(lo > 0 or hi < 0),
    }


def interaction_permutation(df: pd.DataFrame, arch_col: str, codes: list[str],
                            cluster_col: str | None) -> dict:
    """archetype 間の Δ レンジ (max−min) の 4 予測区間平均を統計量とする並べ替え検定.

    cluster_col が指定された場合 (開発者 archetype) はその単位でラベルを保ったまま
    シャッフルする (= cluster→archetype の対応を入れ替える)。None ならペア行単位。
    """
    rng = np.random.default_rng(SEED + 1)

    def stat(labels: np.ndarray) -> float:
        ranges = []
        for h in HORIZONS:
            y = df[f"label_{h}"].astype(int).values
            irl = df[f"irl_{h}"].values
            rf = df[f"rf_{h}"].values
            mask_h = ~(np.isnan(df[f"label_{h}"].values) | np.isnan(irl) | np.isnan(rf))
            ds = []
            for code in codes:
                m = mask_h & (labels == code)
                if m.sum() < 10 or len(np.unique(y[m])) < 2:
                    continue
                ds.append(roc_auc_score(y[m], irl[m]) - roc_auc_score(y[m], rf[m]))
            if len(ds) >= 2:
                ranges.append(max(ds) - min(ds))
        return float(np.mean(ranges)) if ranges else np.nan

    obs_labels = df[arch_col].values.astype(object)
    t_obs = stat(obs_labels)

    if cluster_col is not None:
        # cluster (開発者) ごとに 1 つの archetype を持つ → cluster→archetype を入替
        cl = df[cluster_col].values
        uniq_cl, first_idx = np.unique(cl, return_index=True)
        cl_arch = df[arch_col].values[first_idx]  # 各 cluster の archetype
        pos = {c: i for i, c in enumerate(uniq_cl)}
        row_cluster_pos = np.array([pos[c] for c in cl])
    else:
        row_labels = df[arch_col].values.astype(object)

    perm_stats = np.empty(N_PERM)
    for i in range(N_PERM):
        if cluster_col is not None:
            shuffled_cl_arch = rng.permutation(cl_arch)
            labels = shuffled_cl_arch[row_cluster_pos]
        else:
            labels = rng.permutation(row_labels)
        perm_stats[i] = stat(labels.astype(object))
    perm_stats = perm_stats[~np.isnan(perm_stats)]
    p = float((1 + np.sum(perm_stats >= t_obs)) / (1 + len(perm_stats)))
    return {
        "t_obs_range_mean_pp": round(t_obs * 100, 3),
        "perm_mean_pp": round(float(np.mean(perm_stats)) * 100, 3),
        "perm_p95_pp": round(float(np.percentile(perm_stats, 95)) * 100, 3),
        "p_value": round(p, 5),
        "n_perm": int(len(perm_stats)),
    }


def plot_forest(ci_dev: pd.DataFrame, ci_pair: pd.DataFrame,
                out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True)
    colors = {"0-3m": "#1f77b4", "3-6m": "#ff7f0e",
              "6-9m": "#2ca02c", "9-12m": "#d62728"}

    for ax, ci, codes, name_map, title in [
        (axes[0], ci_dev, CODES, DEV_NAME, "開発者 archetype"),
        (axes[1], ci_pair, ["P1", "P2", "P3", "P4"], PAIR_NAME, "ペア archetype"),
    ]:
        ytick_pos, ytick_lab = [], []
        y = 0.0
        for code in reversed(codes):
            name = name_map[code]
            block_center = y + 1.5 * 0.22 * 1  # 中心 (4 horizon)
            for k, h in enumerate(HORIZONS):
                r = ci[(ci["archetype"] == code) & (ci["horizon"] == h)]
                if r.empty:
                    y += 0.22
                    continue
                r = r.iloc[0]
                yy = y
                lo, hi, d = r["ci_lo_pp"], r["ci_hi_pp"], r["delta_pp"]
                sig = r["sig_ci_excl_0"]
                ax.plot([lo, hi], [yy, yy], color=colors[h], lw=2, alpha=0.9)
                ax.scatter([d], [yy], color=colors[h],
                           s=70 if sig else 36,
                           marker="o" if sig else "o",
                           edgecolor="black" if sig else "none",
                           linewidth=1.2 if sig else 0, zorder=3,
                           label=HORIZON_LABEL_JP[h] if code == codes[-1] else None)
                y += 0.22
            ytick_pos.append(block_center)
            ytick_lab.append(name.replace(" ", "\n", 1))
            y += 0.34  # archetype 間の余白
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_lab, fontsize=10)
        ax.set_xlabel("Δ AUC = IRL − RF (pp)  [95% CI, 塗り＝CIが0を除外]", fontsize=10)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=9, title="予測区間")
    fig.suptitle("archetype × 予測区間の Δ=IRL−RF AUC と開発者クラスタ bootstrap 95% CI",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    pa = pd.read_csv(DATA / "pair_archetype.csv").rename(
        columns={"dominant": "pair_archetype_code"})[
        ["developer", "directory", "pair_archetype_code"]]

    # --- DEV archetype ---
    dev = pred.dropna(subset=["archetype"]).copy()
    print(f"[dev] pairs={len(dev)} devs={dev['developer'].nunique()}")
    ci_dev = compute_ci_table(dev, "dev", "archetype", CODES, DEV_NAME)
    perm_dev = interaction_permutation(dev, "archetype", CODES,
                                       cluster_col="developer")
    print("[dev] interaction:", perm_dev)
    rng_pool = np.random.default_rng(SEED + 2)
    pooled_rows = []
    for code in CODES:
        r = horizon_averaged_delta_ci(dev, "archetype", code, rng_pool)
        r.update({"granularity": "dev", "archetype": code,
                  "archetype_name": DEV_NAME[code]})
        pooled_rows.append(r)

    # --- PAIR archetype ---
    df = pred.merge(pa, on=["developer", "directory"], how="left")
    df = df.dropna(subset=["pair_archetype_code"]).copy()
    # P1-P4 表示コードに変換
    df["pair_archetype"] = df["pair_archetype_code"].map(PMAP)
    print(f"[pair] pairs={len(df)}")
    ci_pair = compute_ci_table(df, "pair", "pair_archetype",
                               ["P1", "P2", "P3", "P4"], PAIR_NAME)
    perm_pair = interaction_permutation(df, "pair_archetype",
                                        ["P1", "P2", "P3", "P4"],
                                        cluster_col=None)
    print("[pair] interaction:", perm_pair)
    for code in ["P1", "P2", "P3", "P4"]:
        r = horizon_averaged_delta_ci(df, "pair_archetype", code, rng_pool)
        r.update({"granularity": "pair", "archetype": code,
                  "archetype_name": PAIR_NAME[code]})
        pooled_rows.append(r)
    pooled = pd.DataFrame(pooled_rows)[
        ["granularity", "archetype", "archetype_name",
         "delta_pp", "ci_lo_pp", "ci_hi_pp", "boot_p", "sig"]]
    pooled.to_csv(OUT / "archetype_delta_ci_pooled.csv", index=False)
    print(f"saved {OUT / 'archetype_delta_ci_pooled.csv'}")
    print("\n=== 4 予測区間平均 Δ (開発者クラスタ bootstrap CI) ===")
    for _, r in pooled.iterrows():
        mark = "★有意" if r["sig"] else ""
        print(f"  {r['granularity']:4s} {r['archetype']} "
              f"Δ̄={r['delta_pp']:+.1f}pp [{r['ci_lo_pp']:+.1f},{r['ci_hi_pp']:+.1f}] "
              f"p={r['boot_p']:.3f} {mark}")

    # --- 保存 ---
    ci_all = pd.concat([ci_dev, ci_pair], ignore_index=True)
    ci_all.to_csv(OUT / "archetype_delta_ci.csv", index=False)
    print(f"saved {OUT / 'archetype_delta_ci.csv'}  ({len(ci_all)} rows)")

    with open(OUT / "archetype_delta_permutation.json", "w") as f:
        json.dump({"dev": perm_dev, "pair": perm_pair,
                   "n_boot": N_BOOT, "n_perm": N_PERM, "seed": SEED}, f,
                  ensure_ascii=False, indent=2)
    print(f"saved {OUT / 'archetype_delta_permutation.json'}")

    plot_forest(ci_dev, ci_pair, OUT / "fig_archetype_delta_ci.pdf")

    # --- サマリ表示 ---
    print("\n=== 有意セル (CI が 0 を除外) ===")
    sig = ci_all[ci_all["sig_ci_excl_0"]]
    for _, r in sig.iterrows():
        side = "IRL優位" if r["delta_pp"] > 0 else "RF優位"
        print(f"  {r['granularity']:4s} {r['archetype']} {r['horizon']:6s} "
              f"Δ={r['delta_pp']:+.1f}pp [{r['ci_lo_pp']:+.1f},{r['ci_hi_pp']:+.1f}] "
              f"p={r['boot_p']:.3f} {side}")
    print(f"\n有意セル数: {len(sig)} / {len(ci_all)}")


if __name__ == "__main__":
    main()

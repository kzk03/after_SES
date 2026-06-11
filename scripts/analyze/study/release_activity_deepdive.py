"""Release rush/lull の深掘り分析。

検証する問い:
  Q1: そもそも release 前後で活動量は増えるか?
  Q2: rush で活動量が増えるなら誰が増やしているか? (archetype 別)
  Q3: H1 補強: A2 の活動シェアは rush で減るか?
       (A2 が薄まる = 他 archetype に比べ A2 の活動が相対的に少なくなる)
  Q4: H2 補強: post-release lull で個人差が拡大するのは A2 だけか?
       (他 archetype と比較して A2 固有の現象か)

入力:
  - data/combined_raw_main32.csv
  - outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/dev_window_pooled_with_alpha.csv

出力:
  - outputs/analysis/release_activity_deepdive/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

YOGA = pd.Timestamp("2022-03-30")
ZED = pd.Timestamp("2022-10-05")
EVAL_START = pd.Timestamp("2022-01-01")
EVAL_END = pd.Timestamp("2023-01-01")
WINDOWS = {
    "0-3m":  (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-04-01")),
    "3-6m":  (pd.Timestamp("2022-04-01"), pd.Timestamp("2022-07-01")),
    "6-9m":  (pd.Timestamp("2022-07-01"), pd.Timestamp("2022-10-01")),
    "9-12m": (pd.Timestamp("2022-10-01"), pd.Timestamp("2023-01-01")),
}
RUSH = ["2022-02", "2022-03", "2022-09", "2022-10"]
LULL = ["2022-05", "2022-06", "2022-11", "2022-12"]


def load(reviews_csv, arche_csv):
    rev = pd.read_csv(reviews_csv, parse_dates=["timestamp"])
    rev = rev[(rev["timestamp"] >= EVAL_START) & (rev["timestamp"] < EVAL_END)].copy()
    rev["label"] = rev["label"].astype(int)
    rev["ym"] = rev["timestamp"].dt.to_period("M").astype(str)
    arche = pd.read_csv(arche_csv)
    dev_arch = (arche.groupby("developer")["dominant"]
                .agg(lambda s: s.value_counts().idxmax())
                .reset_index().rename(columns={"developer": "email", "dominant": "archetype"}))
    return rev.merge(dev_arch, on="email", how="inner")


# ────────────────────────────────────────────────────────────
# Q1, Q2: 月次レビュー量と archetype 構成
# ────────────────────────────────────────────────────────────
def monthly_volume(ev):
    out = ev.groupby(["ym", "archetype"]).size().unstack(fill_value=0)
    out.columns.name = None
    out["TOTAL"] = out.sum(axis=1)
    for arch in ["A1", "A2", "A3", "A4"]:
        if arch in out.columns:
            out[f"{arch}_share"] = (out[arch] / out["TOTAL"]).round(3)
    return out.reset_index()


# ────────────────────────────────────────────────────────────
# Release ±30日の比較
# ────────────────────────────────────────────────────────────
def release_window_compare(ev, release_date, label, days=30):
    pre = ev[(ev["timestamp"] >= release_date - pd.Timedelta(days=days)) &
             (ev["timestamp"] < release_date)]
    post = ev[(ev["timestamp"] >= release_date) &
              (ev["timestamp"] < release_date + pd.Timedelta(days=days))]
    rows = []
    for arch in ["A1", "A2", "A3", "A4"]:
        ap = pre[pre["archetype"] == arch]
        aq = post[post["archetype"] == arch]
        rows.append({
            "release": label, "archetype": arch,
            "n_pre30d": len(ap), "n_post30d": len(aq),
            "delta_volume_pct": ((len(aq) - len(ap)) / len(ap) * 100) if len(ap) else float("nan"),
            "rate_pre": ap["label"].mean() if len(ap) else float("nan"),
            "rate_post": aq["label"].mean() if len(aq) else float("nan"),
            "rate_delta_pp": (aq["label"].mean() - ap["label"].mean()) * 100 if len(ap) and len(aq) else float("nan"),
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# Q3: rush 期での archetype share の変化
# ────────────────────────────────────────────────────────────
def share_in_rush_vs_lull(ev):
    rows = []
    for label, months in [("rush", RUSH), ("lull", LULL)]:
        sub = ev[ev["ym"].isin(months)]
        total = len(sub)
        for arch in ["A1", "A2", "A3", "A4"]:
            n = (sub["archetype"] == arch).sum()
            rows.append({"period": label, "archetype": arch,
                         "n": int(n), "share": n / total if total else float("nan")})
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# Q4: post-release で個人差拡大は A2 固有か (rush vs lull の inter-dev SD)
# ────────────────────────────────────────────────────────────
def individuality_rush_vs_lull(ev, min_events=3):
    rows = []
    for label, months in [("rush", RUSH), ("lull", LULL)]:
        sub = ev[ev["ym"].isin(months)]
        for arch in ["A1", "A2", "A3", "A4"]:
            a = sub[sub["archetype"] == arch]
            per_dev = a.groupby("email")["label"].agg(["mean", "size"]).reset_index()
            per_dev = per_dev[per_dev["size"] >= min_events]
            if len(per_dev) < 5:
                continue
            inter_sd = float(per_dev["mean"].std())

            # dev 内月次 CV
            monthly = a.groupby(["email", "ym"])["label"].mean().reset_index()
            cv_list = []
            for dev, g in monthly.groupby("email"):
                if len(g) < 2: continue
                m = g["label"].mean()
                if m == 0: continue
                cv_list.append(g["label"].std() / m)
            intra_cv = float(np.median(cv_list)) if cv_list else float("nan")

            rows.append({
                "period": label, "archetype": arch,
                "n_dev": int(len(per_dev)),
                "inter_dev_sd": inter_sd,
                "intra_dev_monthly_cv_median": intra_cv,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", type=Path, default=Path("data/combined_raw_main32.csv"))
    ap.add_argument("--archetype-csv", type=Path,
                    default=Path("outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/dev_window_pooled_with_alpha.csv"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/analysis/release_activity_deepdive"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[load]")
    ev = load(args.reviews, args.archetype_csv)
    print(f"  rows={len(ev):,}  devs={ev['email'].nunique():,}")

    # ─── Q1, Q2 ──────────────────────────────────────────
    mv = monthly_volume(ev)
    mv.to_csv(args.output_dir / "monthly_volume.csv", index=False)
    print("\n=== Q1+Q2: 月次レビュー量 (Yoga=3/30, Zed=10/5) ===")
    cols = ["ym", "A1", "A2", "A3", "A4", "TOTAL"]
    print(mv[cols].to_string(index=False))
    print("\n=== archetype share (= n_archetype / TOTAL) ===")
    share_cols = ["ym", "A1_share", "A2_share", "A3_share", "A4_share"]
    print(mv[share_cols].to_string(index=False))

    # ─── release ±30d ────────────────────────────────────
    yoga30 = release_window_compare(ev, YOGA, "Yoga (3/30)")
    zed30 = release_window_compare(ev, ZED, "Zed (10/5)")
    both = pd.concat([yoga30, zed30], ignore_index=True)
    both.to_csv(args.output_dir / "release_pm30d.csv", index=False)
    print("\n=== Release ±30 日の活動量と受諾率 (per archetype) ===")
    print(both.to_string(index=False))

    # ─── Q3 ──────────────────────────────────────────────
    sh = share_in_rush_vs_lull(ev)
    sh.to_csv(args.output_dir / "share_rush_vs_lull.csv", index=False)
    print("\n=== Q3 (H1 補強): rush vs lull の archetype share ===")
    pivot = sh.pivot(index="archetype", columns="period", values="share").round(3)
    pivot["delta_rush_lull"] = (pivot["rush"] - pivot["lull"]).round(3)
    print(pivot.to_string())

    # ─── Q4 ──────────────────────────────────────────────
    ind = individuality_rush_vs_lull(ev)
    ind.to_csv(args.output_dir / "individuality_rush_vs_lull.csv", index=False)
    print("\n=== Q4 (H2 補強): rush vs lull の個人差 (全 archetype) ===")
    print(ind.to_string(index=False))
    print("\n  archetype 別 rush→lull の変化 (inter_dev_sd, intra_dev_cv):")
    print("  期待: A2 だけ lull で SD↑ かつ CV↓ なら H2 は A2 固有現象")
    for arch in ["A1", "A2", "A3", "A4"]:
        sub = ind[ind["archetype"] == arch].set_index("period")
        if "rush" not in sub.index or "lull" not in sub.index:
            continue
        sd_d = sub.loc["lull", "inter_dev_sd"] - sub.loc["rush", "inter_dev_sd"]
        cv_d = sub.loc["lull", "intra_dev_monthly_cv_median"] - sub.loc["rush", "intra_dev_monthly_cv_median"]
        marker = ""
        if sd_d > 0.02 and cv_d < -0.05:
            marker = "  ← H2 (個人差拡大 & 安定化) 支持"
        print(f"    {arch}: ΔSD={sd_d:+.3f}   ΔCV={cv_d:+.3f}{marker}")

    print(f"\n[done] outputs at {args.output_dir}")


if __name__ == "__main__":
    main()

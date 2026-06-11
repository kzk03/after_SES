"""A2 で IRL が 3-6m / 9-12m で特に強い理由について 3 仮説を実データで検証する。

仮説:
  H1: Release rush 期に全 archetype の受諾率が上昇し、A2 の「厳格な査読」特性が
      相対的に薄まる
  H2: Post-release lull 期に A2 が平常モードに戻り、個別軌跡が鮮明になる
      （LSTM hidden state が個人文脈を活用しやすい）
  H3: 窓内に release 境界をまたぐと分布シフトが起き、両モデルがノイズを受ける

入力:
  - data/combined_raw_main32.csv
  - outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/dev_window_pooled_with_alpha.csv

出力:
  - outputs/analysis/a2_release_cycle_hypotheses/
      h1_rush_vs_lull.csv
      h2_individuality_by_window.csv
      h3_boundary_shift.csv
      summary.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# OpenStack release dates (2022 cycle)
YOGA = pd.Timestamp("2022-03-30")
ZED = pd.Timestamp("2022-10-05")

# Evaluation period: 2022-01-01 to 2023-01-01
EVAL_START = pd.Timestamp("2022-01-01")
EVAL_END = pd.Timestamp("2023-01-01")

# Prediction windows (calendar months from t0=2022-01-01)
WINDOWS = {
    "0-3m":  (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-04-01")),
    "3-6m":  (pd.Timestamp("2022-04-01"), pd.Timestamp("2022-07-01")),
    "6-9m":  (pd.Timestamp("2022-07-01"), pd.Timestamp("2022-10-01")),
    "9-12m": (pd.Timestamp("2022-10-01"), pd.Timestamp("2023-01-01")),
}

# Release rush vs lull months
RUSH_MONTHS = [
    pd.Period("2022-02"), pd.Period("2022-03"),  # pre-Yoga
    pd.Period("2022-09"), pd.Period("2022-10"),  # pre-Zed
]
LULL_MONTHS = [
    pd.Period("2022-05"), pd.Period("2022-06"),  # post-Yoga
    pd.Period("2022-11"), pd.Period("2022-12"),  # post-Zed
]


def load_data(reviews_csv: Path, arche_csv: Path) -> pd.DataFrame:
    rev = pd.read_csv(reviews_csv, parse_dates=["timestamp"])
    rev = rev[(rev["timestamp"] >= EVAL_START) & (rev["timestamp"] < EVAL_END)].copy()
    rev["label"] = rev["label"].astype(int)
    rev["ym"] = rev["timestamp"].dt.to_period("M")

    arche = pd.read_csv(arche_csv)
    dev_arch = (
        arche.groupby("developer")["dominant"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={"developer": "email", "dominant": "archetype"})
    )
    return rev.merge(dev_arch, on="email", how="inner")


# ────────────────────────────────────────────────────────────
# H1: release rush 期に全 archetype の受諾率が上昇するか
# ────────────────────────────────────────────────────────────
def test_h1(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ["A1", "A2", "A3", "A4"]:
        sub = ev[ev["archetype"] == arch]
        rush = sub[sub["ym"].isin(RUSH_MONTHS)]
        lull = sub[sub["ym"].isin(LULL_MONTHS)]
        if len(rush) == 0 or len(lull) == 0:
            continue
        rush_rate = rush["label"].mean()
        lull_rate = lull["label"].mean()
        rush_vol = len(rush)
        lull_vol = len(lull)
        rows.append({
            "archetype": arch,
            "n_rush": rush_vol, "n_lull": lull_vol,
            "accept_rate_rush": rush_rate,
            "accept_rate_lull": lull_rate,
            "delta_rush_minus_lull": rush_rate - lull_rate,
            "volume_ratio_rush_over_lull": rush_vol / lull_vol if lull_vol else float("nan"),
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# H2: A2 の個別軌跡が post-release lull 窓で鮮明か
# ────────────────────────────────────────────────────────────
def test_h2(ev: pd.DataFrame, min_events_per_dev: int = 3) -> pd.DataFrame:
    """各予測窓内で A2 dev ごとの受諾率 SD と
       dev 内の月次受諾率変動を計測する。

    指標:
      - per_dev_rate_std: A2 dev 間の平均受諾率の標準偏差 (個人差の大きさ)
      - intra_dev_monthly_cv: 各 A2 dev 内の月次受諾率の変動係数の中央値
        (低いほど dev 内で安定 = 個別軌跡が安定)
    """
    rows = []
    for win, (start, end) in WINDOWS.items():
        for arch in ["A1", "A2", "A3", "A4"]:
            sub = ev[(ev["timestamp"] >= start) & (ev["timestamp"] < end) &
                     (ev["archetype"] == arch)]
            if len(sub) == 0:
                continue
            per_dev = sub.groupby("email")["label"].agg(["mean", "size"]).reset_index()
            per_dev = per_dev[per_dev["size"] >= min_events_per_dev]
            if len(per_dev) < 3:
                continue

            # H2 indicator 1: dev 間 SD (個人差)
            inter_dev_sd = float(per_dev["mean"].std())

            # H2 indicator 2: 月次変動の中央値 (dev 内安定性)
            monthly = sub.groupby(["email", "ym"])["label"].mean().reset_index()
            cv_list = []
            for dev, g in monthly.groupby("email"):
                if len(g) < 2:
                    continue
                m = g["label"].mean()
                if m == 0:
                    continue
                cv_list.append(g["label"].std() / m)
            intra_cv_median = float(np.median(cv_list)) if cv_list else float("nan")

            rows.append({
                "window": win, "archetype": arch,
                "n_dev": int(len(per_dev)), "n_events": int(len(sub)),
                "inter_dev_sd": inter_dev_sd,
                "intra_dev_monthly_cv_median": intra_cv_median,
            })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# H3: 窓内 release 境界をまたぐと分布シフトが起きるか
# ────────────────────────────────────────────────────────────
def test_h3(ev: pd.DataFrame) -> pd.DataFrame:
    """各窓を release 境界の前後で 2 分割し、archetype 別受諾率の差を測る。

    境界をまたがない窓 (3-6m, 9-12m) は窓中央で 2 分割し
    対照群として同じ手法で「自然なドリフト」を測る。
    """
    cases = [
        # (window, split_date, label)
        ("0-3m", YOGA, "Yoga 境界 (0-3m: pre vs post)"),
        ("3-6m", pd.Timestamp("2022-05-15"), "3-6m 窓中央分割 (対照: 境界なし)"),
        ("6-9m", ZED, "Zed 境界 (6-9m: pre vs post)"),
        ("9-12m", pd.Timestamp("2022-11-15"), "9-12m 窓中央分割 (対照: 境界なし)"),
    ]
    rows = []
    for win, split, descr in cases:
        start, end = WINDOWS[win]
        pre = ev[(ev["timestamp"] >= start) & (ev["timestamp"] < split)]
        post = ev[(ev["timestamp"] >= split) & (ev["timestamp"] < end)]
        for arch in ["A1", "A2", "A3", "A4"]:
            a_pre = pre[pre["archetype"] == arch]
            a_post = post[post["archetype"] == arch]
            if len(a_pre) < 10 or len(a_post) < 10:
                continue
            r_pre = a_pre["label"].mean()
            r_post = a_post["label"].mean()
            rows.append({
                "window": win, "split_type": descr, "archetype": arch,
                "n_pre": len(a_pre), "n_post": len(a_post),
                "rate_pre": r_pre, "rate_post": r_post,
                "abs_shift": abs(r_post - r_pre),
                "signed_shift": r_post - r_pre,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", type=Path,
                    default=Path("data/combined_raw_main32.csv"))
    ap.add_argument("--archetype-csv", type=Path,
                    default=Path("outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/dev_window_pooled_with_alpha.csv"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/analysis/a2_release_cycle_hypotheses"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[load] reviews + archetype")
    ev = load_data(args.reviews, args.archetype_csv)
    print(f"  rows={len(ev):,}  devs={ev['email'].nunique():,}  "
          f"archetype-assigned only")

    # ─── H1 ──────────────────────────────────────────────
    h1 = test_h1(ev)
    h1.to_csv(args.output_dir / "h1_rush_vs_lull.csv", index=False)
    print("\n=== H1: Release rush vs Post-release lull (受諾率と活動量) ===")
    print(h1.to_string(index=False))

    # ─── H2 ──────────────────────────────────────────────
    h2 = test_h2(ev)
    h2.to_csv(args.output_dir / "h2_individuality_by_window.csv", index=False)
    print("\n=== H2: 予測窓 x archetype の dev 間 SD / dev 内月次 CV ===")
    print(h2.to_string(index=False))
    print("\n  A2 のみ (post-release 窓で個人差が拡大 or 月次変動が安定するか):")
    print(h2[h2["archetype"] == "A2"].to_string(index=False))

    # ─── H3 ──────────────────────────────────────────────
    h3 = test_h3(ev)
    h3.to_csv(args.output_dir / "h3_boundary_shift.csv", index=False)
    print("\n=== H3: 窓内 release 境界 vs 対照分割の受諾率シフト ===")
    print(h3.to_string(index=False))

    # ─── サマリ判定 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  仮説判定サマリ")
    print("=" * 60)

    # H1: rush で A2 の受諾率上昇が他より顕著か
    if "A2" in h1["archetype"].values:
        a2_rush_delta = h1[h1["archetype"] == "A2"]["delta_rush_minus_lull"].iloc[0]
        others_rush_delta = h1[h1["archetype"] != "A2"]["delta_rush_minus_lull"].mean()
        print(f"\nH1: A2 rush-lull delta = {a2_rush_delta:+.3f},  他 archetype 平均 = {others_rush_delta:+.3f}")
        if a2_rush_delta > others_rush_delta + 0.02:
            print("  → A2 の受諾率は rush 期で他より顕著に上昇 → H1 支持")
        elif abs(a2_rush_delta - others_rush_delta) < 0.02:
            print("  → A2 と他の rush-lull 差は同程度 → H1 弱い")
        else:
            print("  → A2 は rush で他より受諾率が下がる → H1 不支持")

    # H2: A2 で post-release 窓 (3-6m, 9-12m) の個人差が rush 窓 (0-3m, 6-9m) より大きいか
    a2_h2 = h2[h2["archetype"] == "A2"].set_index("window")
    if all(w in a2_h2.index for w in ["0-3m", "3-6m", "6-9m", "9-12m"]):
        rush_sd = (a2_h2.loc["0-3m", "inter_dev_sd"] + a2_h2.loc["6-9m", "inter_dev_sd"]) / 2
        lull_sd = (a2_h2.loc["3-6m", "inter_dev_sd"] + a2_h2.loc["9-12m", "inter_dev_sd"]) / 2
        print(f"\nH2: A2 dev 間 SD  rush 窓平均={rush_sd:.3f}  lull 窓平均={lull_sd:.3f}")
        if lull_sd > rush_sd + 0.01:
            print("  → lull 窓で個人差が拡大 → H2 (個別軌跡の鮮明化) 支持")
        elif abs(lull_sd - rush_sd) < 0.01:
            print("  → 個人差の差は小さい → H2 弱い")
        else:
            print("  → rush 窓の方が個人差大 → H2 不支持")

        rush_cv = (a2_h2.loc["0-3m", "intra_dev_monthly_cv_median"] +
                   a2_h2.loc["6-9m", "intra_dev_monthly_cv_median"]) / 2
        lull_cv = (a2_h2.loc["3-6m", "intra_dev_monthly_cv_median"] +
                   a2_h2.loc["9-12m", "intra_dev_monthly_cv_median"]) / 2
        print(f"     A2 dev 内月次 CV   rush={rush_cv:.3f}  lull={lull_cv:.3f}")
        if lull_cv < rush_cv - 0.05:
            print("     → lull 窓で dev 内が安定 → H2 (個人軌跡の予測しやすさ) 補強")

    # H3: 境界をまたぐ窓 (0-3m, 6-9m) で受諾率シフト > 対照 (3-6m, 9-12m)
    boundary = h3[h3["window"].isin(["0-3m", "6-9m"])]
    control = h3[h3["window"].isin(["3-6m", "9-12m"])]
    if not boundary.empty and not control.empty:
        b_shift = boundary["abs_shift"].mean()
        c_shift = control["abs_shift"].mean()
        print(f"\nH3: 境界窓の平均シフト={b_shift:.3f}  対照窓の平均シフト={c_shift:.3f}")
        if b_shift > c_shift + 0.02:
            print("  → 境界をまたぐと有意に大きなシフト → H3 支持")
        elif abs(b_shift - c_shift) < 0.02:
            print("  → 境界の有無で大差なし → H3 弱い")
        else:
            print("  → 対照窓の方が大きなシフト → H3 不支持")

    with open(args.output_dir / "summary.txt", "w") as f:
        f.write("A2 Release-cycle Hypotheses Verification\n")
        f.write("=" * 60 + "\n\n")
        f.write("=== H1: Rush vs Lull (受諾率) ===\n")
        f.write(h1.to_string(index=False) + "\n\n")
        f.write("=== H2: 窓 x archetype 個人差 ===\n")
        f.write(h2.to_string(index=False) + "\n\n")
        f.write("=== H3: 境界シフト ===\n")
        f.write(h3.to_string(index=False) + "\n")

    print(f"\n[done] outputs at {args.output_dir}")


if __name__ == "__main__":
    main()

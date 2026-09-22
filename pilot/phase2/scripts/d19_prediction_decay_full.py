"""D19: ★SES研究との接続 — AI 参入前に学習した継続予測モデルは、参入後に劣化するか。

元の SES 研究(レビュー継続予測、Gerrit、IRL/RF)の中心的な問いは
「誰がレビューを続け、誰が離脱するか」を予測することだった。
その予測モデルは、AI が開発現場に入ってきた後も通用するのか。

phase2 の発見からは明確な予測が立つ:
  AI レビュー導入後も、人間のレビュー量・人数・PRあたり関与・役割配分は何も変わらなかった(D4, D15)。
  個人レベルでも役割は変わらなかった(S3深掘り)。
  **だとすれば、継続予測モデルも劣化しないはずである。**

これを検証する。どちらの結果でも意味がある。
  - 劣化しない → phase2 の主張が「予測可能性」という別の次元でも裏付けられる(集団・個人・予測の3層で一貫)
  - 劣化する   → phase2 で測れていない何かが変わっている。新しい問いになる

2026-09 に実施した簡易版(`pilot/reports/a1_prediction_decay.md`)は、対象期間が直近12ヶ月しかなく
「AI 参入前」を含まなかったため検証不能だった。全期間(2023-01〜)データで初めて意味のある検証になる。

設計:
  タスク  : (actor, module) が月 t に活動していたとき、t+1..t+HORIZON にも活動するか(二値)
  学習期間: AI 参入前(既定 2023-01〜2024-12)
  テスト  : 参入後を6ヶ月ごとに区切り、AUC の推移を見る
  特徴量  : SES 研究の発想を踏襲(経験・直近活動量・reviewer_owner_ratio・在籍期間・担当モジュール数)
  モデルA : 上記のみ(AI の存在を知らない素朴なモデル)
  モデルB : A + そのモジュールの AI 関与度

実行: PILOT_INTERIM=interim_full PILOT_PROCESSED=processed_full \
      uv run python pilot/phase2/scripts/d19_prediction_decay_full.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import INTERIM_DIR, OUT_DIR, load_repos, repo_dir_name

TRAIN_END = "2024-12"   # ここまでを「AI 参入前」として学習に使う
HORIZON = 3             # 継続とみなす先読み月数
TEST_BLOCK = 6          # テスト期間を何ヶ月ごとに区切るか
MIN_TRAIN = 2000        # 学習サンプルの下限


def build_actor_panel() -> pd.DataFrame:
    """(actor, module, month) の継続予測データセットを作る。"""
    frames = []
    for repo in load_repos():
        ev_path = INTERIM_DIR / repo_dir_name(repo) / "events.parquet"
        if not ev_path.exists():
            continue
        ev = pd.read_parquet(ev_path)
        ev = ev[ev["stage"].isin(["G", "E"]) & (ev["actor_category"] == "human")]
        if ev.empty:
            continue

        months = sorted(ev["month"].unique())
        midx = {m: i for i, m in enumerate(months)}

        # (actor, module, month) の活動量と、E/G の内訳
        g = ev.groupby(["actor_login", "module_id", "month"]).agg(
            n_events=("stage", "size"),
            n_E=("stage", lambda s: (s == "E").sum()),
        ).reset_index()
        g["month_idx"] = g["month"].map(midx)

        # actor 単位の月次集計(在籍期間・担当モジュール数・reviewer_ratio 用)
        a = ev.groupby(["actor_login", "month"]).agg(
            a_events=("stage", "size"),
            a_E=("stage", lambda s: (s == "E").sum()),
            a_modules=("module_id", "nunique"),
        ).reset_index()
        a["month_idx"] = a["month"].map(midx)
        a["reviewer_ratio"] = a["a_E"] / a["a_events"]   # SES研究の reviewer_owner_ratio と同じ定義
        first_seen = a.groupby("actor_login")["month_idx"].min().rename("first_idx")

        # 活動集合(継続判定用)
        active = {(r.actor_login, r.module_id, r.month_idx) for r in g.itertuples()}

        g = g.merge(a[["actor_login", "month_idx", "reviewer_ratio", "a_modules", "a_events"]],
                    on=["actor_login", "month_idx"], how="left")
        g = g.join(first_seen, on="actor_login")

        # 累積活動月数(その時点までに何ヶ月活動したか)
        g = g.sort_values(["actor_login", "module_id", "month_idx"])
        g["n_months_active_so_far"] = g.groupby(["actor_login", "module_id"]).cumcount() + 1

        n_months = len(months)
        rows = []
        for r in g.itertuples():
            if r.month_idx + HORIZON >= n_months:
                continue  # 先読みできない末尾は使わない
            cont = any((r.actor_login, r.module_id, r.month_idx + j) in active
                       for j in range(1, HORIZON + 1))
            rows.append({
                "repo": repo,
                "module_id_global": f"{repo}::{r.module_id}",
                "month": r.month,
                "n_months_active_so_far": r.n_months_active_so_far,
                "log_recent": np.log1p(r.n_events),
                "reviewer_ratio": r.reviewer_ratio,
                "log_n_modules": np.log1p(r.a_modules),
                "tenure_months": r.month_idx - r.first_idx,
                "continues": int(cont),
            })
        if rows:
            frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def attach_ai_share() -> pd.DataFrame:
    """module x month の AI 関与度(A')を panel.parquet から集める。"""
    from common import PROCESSED_DIR
    frames = []
    for repo in load_repos():
        p = PROCESSED_DIR / repo_dir_name(repo) / "panel.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)[["module_id", "month", "A_prime"]]
        d["module_id_global"] = repo + "::" + d["module_id"]
        frames.append(d[["module_id_global", "month", "A_prime"]])
    return pd.concat(frames, ignore_index=True)


FEATS_A = ["n_months_active_so_far", "log_recent", "reviewer_ratio", "log_n_modules", "tenure_months"]
FEATS_B = FEATS_A + ["A_prime"]


def evaluate(train: pd.DataFrame, blocks: list[tuple[str, pd.DataFrame]], feats: list[str]) -> dict:
    d_tr = train.dropna(subset=feats + ["continues"])
    sc = StandardScaler().fit(d_tr[feats])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(d_tr[feats]), d_tr["continues"])

    res = {"n_train": int(len(d_tr)),
           "train_auc": float(roc_auc_score(d_tr["continues"], clf.predict_proba(sc.transform(d_tr[feats]))[:, 1])),
           "coef": dict(zip(feats, clf.coef_[0].tolist())),
           "blocks": []}
    for label, te in blocks:
        d_te = te.dropna(subset=feats + ["continues"])
        if len(d_te) < 200 or d_te["continues"].nunique() < 2:
            res["blocks"].append({"period": label, "n": int(len(d_te)), "auc": None})
            continue
        auc = roc_auc_score(d_te["continues"], clf.predict_proba(sc.transform(d_te[feats]))[:, 1])
        res["blocks"].append({
            "period": label, "n": int(len(d_te)), "auc": float(auc),
            "continues_rate": float(d_te["continues"].mean()),
        })
    return res


def main() -> None:
    print("データセットを構築しています...")
    df = build_actor_panel()
    ai = attach_ai_share()
    df = df.merge(ai, on=["module_id_global", "month"], how="left")
    df["A_prime"] = df["A_prime"].fillna(0.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "d19_dataset.parquet", index=False)

    months = sorted(df["month"].unique())
    print(f"観測期間: {months[0]} 〜 {months[-1]}  n={len(df):,}  "
          f"actor x module x month, {df['repo'].nunique()} リポジトリ")

    if months[0] > "2023-06":
        print("\n!! 学習期間に AI 参入前が十分含まれていない可能性があります。")
        print(f"   データ開始が {months[0]} です。全期間データで実行してください。")

    train = df[df["month"] <= TRAIN_END]
    test_months = [m for m in months if m > TRAIN_END]
    if len(train) < MIN_TRAIN or not test_months:
        print(f"\n学習サンプル {len(train)} 件・テスト月 {len(test_months)} で、検証には不足しています。")
        return

    blocks = []
    for i in range(0, len(test_months), TEST_BLOCK):
        ms = test_months[i:i + TEST_BLOCK]
        blocks.append((f"{ms[0]}〜{ms[-1]}", df[df["month"].isin(ms)]))

    # 参考: 各テスト期間の AI レビュー関与度の水準(環境がどれだけ変わったか)
    ai_level = {lab: float(te["A_prime"].mean()) for lab, te in blocks}
    ai_level_train = float(train["A_prime"].mean())

    print(f"\n学習期間: {months[0]}〜{TRAIN_END}  n={len(train):,}  平均 A'={ai_level_train:.4f}")
    print("テスト期間:")
    for lab, te in blocks:
        print(f"  {lab}  n={len(te):,}  平均 A'={ai_level[lab]:.4f}")

    out = {"train_end": TRAIN_END, "horizon": HORIZON,
           "ai_level_train": ai_level_train, "ai_level_test": ai_level, "models": {}}

    for name, feats in [("A(AI を知らない素朴なモデル)", FEATS_A), ("B(A + モジュールの AI 関与度)", FEATS_B)]:
        r = evaluate(train, blocks, feats)
        out["models"][name] = r
        print(f"\n=== モデル{name} ===")
        print(f"  学習 AUC (in-sample): {r['train_auc']:.4f}  (n={r['n_train']:,})")
        base = None
        for b in r["blocks"]:
            if b["auc"] is None:
                print(f"  {b['period']}  n={b['n']:,}  AUC=(データ不足)")
                continue
            if base is None:
                base = b["auc"]
            print(f"  {b['period']}  n={b['n']:>8,}  AUC={b['auc']:.4f}  "
                  f"(学習時との差 {b['auc'] - r['train_auc']:+.4f})")
        print("  係数:", {k: round(v, 3) for k, v in r["coef"].items()})

    # 判定
    ra = out["models"]["A(AI を知らない素朴なモデル)"]
    aucs = [b["auc"] for b in ra["blocks"] if b["auc"] is not None]
    if len(aucs) >= 2:
        drift = aucs[-1] - aucs[0]
        out["auc_drift_first_to_last"] = float(drift)
        print(f"\n★ モデルA の AUC 推移: 最初のテスト期間 {aucs[0]:.4f} → 最後 {aucs[-1]:.4f} ({drift:+.4f})")
        if drift < -0.03:
            print("  → AI 浸透とともに予測性能が劣化している。phase2 の『何も変わっていない』と整合しない")
        elif abs(drift) <= 0.03:
            print("  → 劣化は見られない。phase2 の主張(人間の行動は変わっていない)と整合する")
        else:
            print("  → むしろ改善している。別の要因(全体的な活動の落ち着き等)を検討する必要がある")

    (OUT_DIR / "d19_prediction_decay.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'd19_prediction_decay.json'}")


if __name__ == "__main__":
    main()

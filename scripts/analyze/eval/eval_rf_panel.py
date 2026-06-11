"""月次パネル RF_Dir 評価 (IRL 準拠の3年分累積データで RF を学習).

通常の RF_Dir は train_end の単一スナップショットだけ使うが、本スクリプトは
各月 m ∈ [train_start, train_end] で `(dev, dir)` 特徴量を再計算して学習データ
に積む (cumulative window: [train_start, m])。
これにより IRL と同じ情報量で RF を学習できる。

Usage:
    uv run python scripts/analyze/eval/eval_rf_panel.py \
        --data data/combined_raw_main32.csv \
        --raw-json data/raw_json/openstack__*.json \
        --train-start 2019-01-01 \
        --train-end 2022-01-01 \
        --prediction-time 2023-01-01 \
        --future-start-months 0 \
        --delta-months 3 \
        --month-step 1 \
        --output-dir outputs/local_rf_panel/fs_0_to_3m
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from review_predictor.IRL.features.common_features import (
    FEATURE_NAMES_WITH_PATH,
    extract_common_features,
)
from review_predictor.IRL.features.path_features import (
    PATH_FEATURE_NAMES,
    PathFeatureExtractor,
    attach_dirs_to_df,
    load_change_dir_map_multi,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 777  # IRL の RANDOM_SEED=777 に統一


def extract_one_month(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    m_end: pd.Timestamp,
    label_start: pd.Timestamp,
    label_end: pd.Timestamp,
    path_extractor: PathFeatureExtractor,
    dirs_column: str = "dirs",
) -> List[Dict]:
    """1ヶ月分のパネルデータを抽出 ((dev, dir) 単位)."""
    feature_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < m_end)
    feat_df = df[feature_mask]
    if len(feat_df) == 0:
        return []
    active_emails = feat_df["email"].unique()

    label_mask = (df["timestamp"] >= label_start) & (df["timestamp"] < label_end)
    label_df = df[label_mask]

    rows: List[Dict] = []
    for email in active_emails:
        dev_feat = feat_df[feat_df["email"] == email]
        dev_dirs: set = set()
        for dirs in dev_feat[dirs_column]:
            if dirs:
                dev_dirs.update(d for d in dirs if d != ".")
        if not dev_dirs:
            continue

        common_feats = extract_common_features(
            df, email, train_start, m_end, normalize=False
        )

        for directory in dev_dirs:
            dev_eval = label_df[
                (label_df["email"] == email)
                & (label_df[dirs_column].map(
                    lambda ds: directory in ds if ds else False
                ))
            ]
            label = 1 if len(dev_eval) > 0 and (dev_eval["label"] == 1).any() else 0

            pf = path_extractor.compute(
                email, frozenset({directory}), m_end.to_pydatetime()
            )
            path_feats = {
                name: float(val) for name, val in zip(PATH_FEATURE_NAMES, pf)
            }

            row = {**common_feats, **path_feats}
            row["label"] = label
            row["email"] = email
            row["directory"] = directory
            row["month"] = m_end
            rows.append(row)
    return rows


def build_monthly_panel(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    future_start_months: int,
    delta_months: int,
    path_extractor: PathFeatureExtractor,
    month_step: int = 1,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """月次パネルを構築. label_end が train_end を超えないようクリップ."""
    months = pd.date_range(train_start, train_end, freq="MS").tolist()
    months = months[::month_step]

    valid_months: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for m_end in months:
        label_start = m_end + pd.DateOffset(months=future_start_months)
        label_end = label_start + pd.DateOffset(months=delta_months)
        if label_end > train_end:
            # ラベル窓が train_end を超える月はスキップ (リーク防止)
            continue
        valid_months.append((m_end, label_start, label_end))

    logger.info(
        f"月次パネル構築: {len(valid_months)} 月分 ({months[0].date()} 〜 {valid_months[-1][0].date() if valid_months else 'N/A'})"
    )

    # 月ごとに並列抽出
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(extract_one_month)(
            df, train_start, m_end, ls, le, path_extractor
        )
        for m_end, ls, le in valid_months
    )

    all_rows: List[Dict] = []
    for rows in results:
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame(
            columns=FEATURE_NAMES_WITH_PATH + ["label", "email", "directory", "month"]
        )

    panel = pd.DataFrame(all_rows)
    logger.info(
        f"パネル完成: {len(panel)} 行, "
        f"positive={int(panel['label'].sum())}, "
        f"negative={int((1 - panel['label']).sum())}, "
        f"unique devs={panel['email'].nunique()}, "
        f"unique dirs={panel['directory'].nunique()}, "
        f"unique months={panel['month'].nunique()}"
    )
    return panel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--raw-json", type=str, nargs="+", required=True)
    p.add_argument("--train-start", type=str, default="2019-01-01")
    p.add_argument("--train-end", type=str, default="2022-01-01")
    p.add_argument("--prediction-time", type=str, default="2023-01-01")
    p.add_argument("--future-start-months", type=int, default=0)
    p.add_argument("--delta-months", type=int, default=3)
    p.add_argument("--window-days", type=int, default=180,
                   help="推論時の特徴量窓 (default 180日)")
    p.add_argument("--dir-depth", type=int, default=2)
    p.add_argument("--month-step", type=int, default=1,
                   help="何ヶ月おきにパネルサンプリングするか (1=毎月)")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── データ読み込み ──
    logger.info(f"データ読み込み: {args.data}")
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── path_extractor 構築 ──
    logger.info(f"change_dir_map 構築: {len(args.raw_json)} JSON files")
    cdm = load_change_dir_map_multi(args.raw_json, depth=args.dir_depth)
    df = attach_dirs_to_df(df, cdm)
    has_dirs = df["dirs"].map(len).gt(0).sum()
    logger.info(f"dirs 列付与: {has_dirs}/{len(df)} 行に dir 情報")

    path_extractor = PathFeatureExtractor(df, window_days=args.window_days)

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    prediction_time = pd.Timestamp(args.prediction_time)

    # ── 月次パネル構築 (キャッシュあれば再利用) ──
    cache_dir = Path("outputs/local_rf_panel/_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = (
        f"panel_ts{args.train_start}_te{args.train_end}"
        f"_fs{args.future_start_months}_dm{args.delta_months}"
        f"_ms{args.month_step}.pkl"
    )
    cache_path = cache_dir / cache_key

    if cache_path.exists():
        logger.info(f"パネルキャッシュ読み込み: {cache_path}")
        panel = pd.read_pickle(cache_path)
        logger.info(f"キャッシュから復元: {len(panel)} 行")
    else:
        panel = build_monthly_panel(
            df=df,
            train_start=train_start,
            train_end=train_end,
            future_start_months=args.future_start_months,
            delta_months=args.delta_months,
            path_extractor=path_extractor,
            month_step=args.month_step,
            n_jobs=args.n_jobs,
        )
        if len(panel) >= 100:
            panel.to_pickle(cache_path)
            logger.info(f"パネルをキャッシュ保存: {cache_path}")

    if len(panel) < 100 or panel["label"].nunique() < 2:
        logger.error("パネルが空 or 単一クラスのみ、学習中止")
        return

    X_train = panel[FEATURE_NAMES_WITH_PATH].values.astype(float)
    y_train = panel["label"].values.astype(int)
    logger.info(f"学習: X={X_train.shape}, y mean={y_train.mean():.3f}")

    # ── RF 学習 ──
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    logger.info("RF 学習完了")

    # ── Feature importance ──
    importance = {
        name: float(val)
        for name, val in zip(FEATURE_NAMES_WITH_PATH, clf.feature_importances_)
    }
    imp_path = args.output_dir / "rf_panel_feature_importance.json"
    with open(imp_path, "w") as f:
        json.dump({"feature_importance": importance}, f, indent=2, ensure_ascii=False)
    logger.info(f"feature importance 保存: {imp_path}")

    # ── 推論 (prediction_time 時点) ──
    feature_start_inf = prediction_time - pd.Timedelta(days=args.window_days)
    label_start_inf = prediction_time + pd.DateOffset(months=args.future_start_months)
    label_end_inf = label_start_inf + pd.DateOffset(months=args.delta_months)

    logger.info(
        f"推論窓: feat [{feature_start_inf.date()}, {prediction_time.date()}), "
        f"label [{label_start_inf.date()}, {label_end_inf.date()})"
    )

    # 推論対象 (dev, dir) ペア: prediction_time までに dev が触った全 dir
    feat_mask_inf = (df["timestamp"] >= feature_start_inf) & (df["timestamp"] < prediction_time)
    feat_inf = df[feat_mask_inf]
    label_mask_inf = (df["timestamp"] >= label_start_inf) & (df["timestamp"] < label_end_inf)
    label_inf = df[label_mask_inf]

    inf_pairs: List[Tuple[str, str, int]] = []
    for email, dev_df in feat_inf.groupby("email"):
        dev_dirs: set = set()
        for dirs in dev_df["dirs"]:
            if dirs:
                dev_dirs.update(d for d in dirs if d != ".")
        for directory in dev_dirs:
            # 実測ラベル
            dev_label = label_inf[
                (label_inf["email"] == email)
                & (label_inf["dirs"].map(
                    lambda ds: directory in ds if ds else False
                ))
            ]
            actual = 1 if len(dev_label) > 0 and (dev_label["label"] == 1).any() else 0
            inf_pairs.append((email, directory, actual))

    logger.info(f"推論対象ペア数: {len(inf_pairs)}")

    # ペア単位の delayed をやめてバッチ単位に。df を pickle する回数を激減。
    BATCH_SIZE = 200
    pair_only = [(e, d) for e, d, _ in inf_pairs]
    batches = [pair_only[i : i + BATCH_SIZE] for i in range(0, len(pair_only), BATCH_SIZE)]
    logger.info(f"バッチ化: {len(batches)} バッチ × {BATCH_SIZE} ペア")

    def _predict_batch(batch_pairs: List[Tuple[str, str]]) -> List[float]:
        # email ごとに common_feats を 1 度だけ計算してキャッシュ
        common_cache: Dict[str, Dict] = {}
        rows: List[List[float]] = []
        for email, directory in batch_pairs:
            if email not in common_cache:
                common_cache[email] = extract_common_features(
                    df, email, feature_start_inf, prediction_time, normalize=False
                )
            common_feats = common_cache[email]
            pf = path_extractor.compute(
                email, frozenset({directory}), prediction_time.to_pydatetime()
            )
            path_feats = {
                name: float(val) for name, val in zip(PATH_FEATURE_NAMES, pf)
            }
            feat_row = {**common_feats, **path_feats}
            rows.append([feat_row.get(f, 0.0) for f in FEATURE_NAMES_WITH_PATH])
        X = np.array(rows, dtype=np.float64)
        return clf.predict_proba(X)[:, 1].tolist()

    batch_results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(_predict_batch)(b) for b in batches
    )
    probs: List[float] = []
    for br in batch_results:
        probs.extend(br)

    pred_df = pd.DataFrame({
        "email": [e for e, _, _ in inf_pairs],
        "directory": [d for _, d, _ in inf_pairs],
        "rf_panel_prob": probs,
        "actual": [a for _, _, a in inf_pairs],
    })
    pred_path = args.output_dir / "rf_panel_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"予測 CSV 保存: {pred_path}")

    # ── ペア単位の AUC ──
    if pred_df["actual"].nunique() >= 2:
        auc_roc = float(roc_auc_score(pred_df["actual"], pred_df["rf_panel_prob"]))
        auc_pr = float(average_precision_score(pred_df["actual"], pred_df["rf_panel_prob"]))
        # F1 最適閾値で precision / recall / F1 を計算
        # precision_recall_curve の返り値: prec/rec は len(thr)+1, 末尾 (prec=1, rec=0) は閾値なし
        prec_arr, rec_arr, thr_arr = precision_recall_curve(
            pred_df["actual"], pred_df["rf_panel_prob"]
        )
        denom = prec_arr + rec_arr
        with np.errstate(invalid="ignore", divide="ignore"):
            f1_arr = np.where(denom > 0, 2 * prec_arr * rec_arr / denom, 0.0)
        # 閾値が定義される indices (= 末尾を除く) で F1 最大を探す
        if len(thr_arr) > 0:
            valid_f1 = f1_arr[:-1]
            best_idx = int(np.argmax(valid_f1))
            clf_threshold = float(thr_arr[best_idx])
            clf_precision = float(prec_arr[best_idx])
            clf_recall = float(rec_arr[best_idx])
            clf_f1 = float(f1_arr[best_idx])
        else:
            clf_threshold = float("nan")
            clf_precision = float("nan")
            clf_recall = float("nan")
            clf_f1 = float("nan")
    else:
        auc_roc = float("nan")
        auc_pr = float("nan")
        clf_threshold = float("nan")
        clf_precision = float("nan")
        clf_recall = float("nan")
        clf_f1 = float("nan")

    base_rate = float(pred_df["actual"].mean())

    metrics = {
        "n_pairs": int(len(pred_df)),
        "positive_count": int(pred_df["actual"].sum()),
        "base_rate": base_rate,
        "pair_auc_roc": auc_roc,
        "pair_auc_pr": auc_pr,
        "clf_threshold": clf_threshold,
        "clf_precision": clf_precision,
        "clf_recall": clf_recall,
        "clf_f1": clf_f1,
        "train_panel_size": int(len(panel)),
        "train_positive_count": int(y_train.sum()),
        "train_unique_devs": int(panel["email"].nunique()),
        "train_unique_dirs": int(panel["directory"].nunique()),
        "train_unique_months": int(panel["month"].nunique()),
        "config": {
            "train_start": args.train_start,
            "train_end": args.train_end,
            "prediction_time": args.prediction_time,
            "future_start_months": args.future_start_months,
            "delta_months": args.delta_months,
            "window_days": args.window_days,
            "month_step": args.month_step,
            "n_estimators": args.n_estimators,
        },
    }
    metrics_path = args.output_dir / "summary_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"summary_metrics 保存: {metrics_path}")

    logger.info(
        f"=== 結果: pair_AUC_ROC={auc_roc:.4f}, "
        f"pair_AUC_PR={auc_pr:.4f}, P={clf_precision:.4f}, R={clf_recall:.4f}, "
        f"F1={clf_f1:.4f} @thr={clf_threshold:.4f}, "
        f"base_rate={base_rate:.4f}, "
        f"n={len(pred_df)} (pos={pred_df['actual'].sum()}) ==="
    )


if __name__ == "__main__":
    main()

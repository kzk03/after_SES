# 最終提出版.pdf（SES論文）再現手順

対象: `writing/最終提出版.pdf`（生成元 TeX: `writing/SES.tex`）。
本リポジトリは 2026-06-12 に「この PDF の完全再現チェーン」だけを残す形に整理した。
削除した分析・実験一式は別リポジトリ（フル版）で管理。

## 論文の数値・図の出所（検証済み対応表）

| 論文要素 | 出所 |
|---|---|
| 表 `tab:rq1`（ペア単位 AUC, n=4,534）IRL 列 0.826/0.813/0.751/0.754 | `outputs/dynstartB_ep150`（IRL, DYNAMIC_START=1, ep150）。集計値は `writing/dynstartB_irl_vs_rf_33m.md` の headline 表（B定義ラベルで aligned 4,534 ペア再計算） |
| 表 `tab:rq1` RF 列 0.791/0.812/0.792/0.806 | `outputs/dynstart_rf_1snap_s777`（**RF_1snap, seed777**。`dynstart_rf_panel_s777` ではない点に注意） |
| 表 `tab:rq1-1`（開発者単位 AUC, N=436） | `writing/revision_proposal.md` の表（開発者ごと max スコア集約、同じ 2 出力から計算。計算系譜は `scripts/analyze/study/per_dev_auc_inclusive.py`） |
| 表 `tab:rq3`（アンサンブル, Ens 0.838/0.842/0.799/0.808） | `writing/dynstartB_irl_vs_rf_33m.md` の Ens_1snap 列＝IRL と RF_1snap の単純平均。CI 含む表全体は `writing/revision_proposal.md` L165-184 |
| 表 `tab:pair-gain` / `tab:dev-gain`（原型別アンサンブル利得） | `writing/revision_proposal.md` L185-217（Ens−最良単独, pp） |
| 図 fig_dev_elbow / fig_pair_elbow | `scripts/analyze/study/render_elbow_figures.py` ← `outputs/analysis/archetype_{irl_rf,pair_level}_volat_ep150_coldonly/rss_elbow.csv` |
| 図 fig_dev_archetype_profile / fig_pair_archetype_profile | `scripts/analyze/study/archetype_profile_eugster.py` ← 同上 dirs の `k4_archetype_profile.csv` 等 |
| 図 fig_model / fig_label3 | 手作り図（生成スクリプトなし）。`writing/fig/` のファイルが原本、消すと再現不能 |

アーキタイプ系の入力 `outputs/analysis/archetype_*_volat_ep150_coldonly` は
`outputs/main32_mce_volat_ep150`（cold IRL volat ep150 + RF_6m seed42）から生成。

## 再現チェーン（学習から）

前提: `uv sync`（`pyproject.toml`）。IRL 学習は GPU 推奨（`DEVICE=cuda`、ローカルは mps/cpu）。

1. **データ収集**（再収集する場合。収集済みデータは `data/` に保持済み）
   `scripts/pipeline/collect_service_teams.sh` → `data/raw_json/openstack__*.json`（main32 の 32 ファイル）
   → `scripts/pipeline/build_dataset.py` 経由で `data/combined_raw_main32.csv`
   （ラベル定義: response_window 内にコメント/投票で反応 = 1。`build_dataset.py` L307/L375-395）
2. **補助特徴量**
   - co-change graph: `experiments/dependency_analysis/01b_cochange_graph_main32.py`
     （入力 = `data/raw_json/openstack__*.json`）→ `results/hub_scores_main32.csv`,
     `results/cochange_neighbors_main32.csv`（IRL/RF の path 特徴 6 次元のうち 2 つ。
     IRL モデルは state_dim=32 = state26 + path6 でこれらに依存）
   - bot 除外は `build_dataset.py` 内蔵のパターンマッチ（論文 4 章の記述と一致）。
     プロフィール特徴 5 次元（dev_history）は最終モデル（32 次元）では未使用のため削除済み
3. **IRL 主結果（表2 IRL 列）**
   ```bash
   PYTHONHASHSEED=0 DYNAMIC_START=1 EPOCHS=150 \
     HUB_SCORES_CSV=experiments/dependency_analysis/results/hub_scores_main32.csv \
     COCHANGE_NEIGHBORS_CSV=experiments/dependency_analysis/results/cochange_neighbors_main32.csv \
     bash scripts/variant/run_mce_irl_variant_single.sh 0 lstm_baseline outputs/dynstartB_ep150 0
   ```
   train_start は horizon 別に 2019-01/2018-10/2018-07/2018-04（33ヶ月等化）、eval cutoff 2023-01-01。
4. **RF 主結果（表2 RF 列）**
   ```bash
   SEED=777 OUTBASE=outputs/dynstart_rf_1snap_s777 \
     bash scripts/local/run_rf_dynstart_1snap_local.sh
   ```
5. **表2/表3 の集計**: 両出力の cold 予測（`irl_dir_prob` / `rf_dir_prob`）を
   prediction_time=2023-01-01, pool=180 日で align（4,534 ペア）、B定義ラベルを
   `data/combined_raw_main32.csv` から再計算して AUC-ROC。手順詳細と全数値は
   `writing/dynstartB_irl_vs_rf_33m.md`（ペア単位）と `writing/revision_proposal.md`（開発者単位）。
6. **アーキタイプ図（elbow / profile）**
   ```bash
   bash scripts/variant/start_all_volat.sh   # EPOCHS=150 → outputs/main32_mce_volat_ep150
   uv run python scripts/analyze/study/archetype_irl_rf.py \
     --cold-dir outputs/main32_mce_volat_ep150 --warm-dir outputs/main32_mce_volat_ep150 \
     --k-grid 2 3 4 5 6 --output-dir outputs/analysis/archetype_irl_rf_volat_ep150_coldonly
   uv run python scripts/analyze/study/archetype_pair_level.py \
     --cold-dir outputs/main32_mce_volat_ep150 --warm-dir outputs/main32_mce_volat_ep150 \
     --k-grid 2 3 4 5 6 --output-dir outputs/analysis/archetype_pair_level_volat_ep150_coldonly
   uv run python scripts/analyze/study/render_elbow_figures.py        # → writing/fig/fig_{dev,pair}_elbow.pdf
   uv run python scripts/analyze/study/archetype_profile_eugster.py   # dev profile
   uv run python scripts/analyze/study/archetype_profile_eugster.py --pair  # pair profile
   ```
7. **PDF ビルド**: `cd writing && latexmk SES.tex`（ルートの `latexmkrc` 参照、pLaTeX 系）。
   参考文献は `writing/SES.bbl` / `writing/Hashimoto.bib`。

## 注意

- `writing/最終提出版.pdf`・`data/`・このファイルは git 未追跡（削除すると復元不能）。
  それ以外の削除済みファイルは init コミット（58eaa63 時点）から `git checkout` で復元可能。
- 表2の生 `summary_metrics.json`（diagonal IRL_Dir 0.786 等）は表の値と一致しない。
  表は cold 予測の aligned 再集計（手順 5）後の値。

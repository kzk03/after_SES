# RL 実験 → 評価 → 考察 → 論文執筆 パイプライン

作成日: 2026-07-04
対象: オフライン RL (BC→IQL) によるレビュー継続予測・SANER 2027 投稿計画
関連: [rl_redesign_plan.md](rl_redesign_plan.md)（設計判断の記録）、writing/rl_roadmap.md（詳細設計）

```
Stage 1          Stage 2            Stage 3              Stage 4
実験 (sweep) ─→ 評価・判断ゲート ─→ 考察 (RQ2 分析) ─→ 論文執筆
run_rl_sweep     summarize +         analyze_value_       図表マッピング
                 compare_methods     drivers ほか          + threats
```

各 Stage は前の Stage の出力だけに依存する。コード変更なしで
コマンドを順に実行すれば論文の図表の素材まで揃う設計。

---

## Stage 1: 実験

### 1.1 実行コマンド（サーバ側）

```bash
# 前提: IRL モデル (outputs/main32_mce_volat_ep150/<variant>/train_<t0>m/) と
#       軌跡キャッシュ (outputs/mce_irl_trajectory_cache/mce_traj_<t0>.pkl)
#       キャッシュが無ければ先に:
#       bash scripts/rl/run_rl_pipeline.sh 0-3 lstm_baseline cpu

# 主実験: event+月次 × λ∈{0,0.1,0.3,0.5,1.0}、dev holdout 10%、集約まで自動
bash scripts/rl/run_rl_sweep.sh 0-3 lstm_baseline cpu

# γ 感度を追加で見る場合 (λ は主実験の最良値に固定して γ を振る例)
LAMBDAS="0.3" GAMMAS="0.9 0.95 0.99" bash scripts/rl/run_rl_sweep.sh 0-3 lstm_baseline cpu

# 複数 seed (統計的頑健性。§4 の分散報告用)
for s in 1 2 3 4 5; do SEED=$s LAMBDAS="0.3" bash scripts/rl/run_rl_sweep.sh 0-3 lstm_baseline cpu; done
#   ※ 現状 seed 別の出力ディレクトリ分けは未対応 (同一 run_dir を上書きする)。
#     複数 seed を保存する場合は SWEEP_ROOT を seed 別に変えるか、
#     run_rl_sweep.sh に seed をディレクトリ名へ含める1行修正を入れる。

# 複数 t0 (rolling origin。IRL 論文 threats への回答)
for w in 0-3 3-6 6-9 9-12; do bash scripts/rl/run_rl_sweep.sh $w lstm_baseline cpu; done
```

### 1.2 実行直後のサニティチェック

| 確認項目 | 見る場所 | 問題があるときのサイン |
|----------|---------|----------------------|
| test が分離されたか | `summary_metrics.json` の `_meta.is_held_out == true` | false → test split が空 (§7.5 の抽出期間問題) |
| test 件数 | `_meta.test_n` が十分 (数百以上) | 少なすぎると全比較が不安定 |
| 過学習 | train 用 eval と test の AUC 差 | 差が大きい → IQL_EPOCHS/τ/β を調整 |
| 報酬分布 | ビルドログの「報酬分布」「R_θ 単体分布」 | R_θ が退化 (std≈0) → IRL モデル/次元指定ミス |
| action バランス | ビルドログの action=1 割合 | 極端 (99%等) → 行動定義のバグを疑う |
| dev holdout | `*_devholdout.parquet` の存在と件数 | 無い → HOLDOUT_DEVS_RATIO 未設定 |

---

## Stage 2: 評価と判断ゲート

### 2.1 集約結果を読む

```bash
# sweep 実行時に自動生成される。手動再集約は:
uv run python scripts/rl/summarize_rl_results.py outputs/rl/sweep \
    --output-csv outputs/rl/sweep/summary.csv --output-md outputs/rl/sweep/summary.md
cat outputs/rl/sweep/summary.md
```

### 2.2 統計的検定（主張に有意性を付ける）

```bash
# 主仮説「IQL/V が IRL・教師ありより良い」を paired bootstrap で検定
uv run python scripts/rl/compare_methods_stats.py \
    outputs/rl/sweep/lstm_baseline_0-3m_event/lam0.3_gam0.99/eval/pair_predictions.csv \
    --target step_label --baseline score_irl_pi \
    --output-dir outputs/rl/analysis/event_lam0.3

# dev 単位ラベル・承諾予測軸でも同様に
#   --target step_label_dev / --target monthly_accept
# 教師あり基準との比較は --baseline score_rf_supervised
```

出力: 各手法の AUC 95%CI、baseline との ΔAUC の CI と bootstrap p 値
（IRL 論文の表8 と同じ流儀。delta_ci が 0 を跨がなければ有意と書ける）。

### 2.3 判断ゲート（結果を見て決めること）

| ゲート | 見る数値 | 判断 |
|--------|---------|------|
| G1: IRL を残すか | λ=0 と λ>0 の `RL_iql_*` AUC 差 | 差が無い → RL 単独の物語に簡素化（IRL は比較手法へ格下げ）。差がある → IRL→RL の2段構成を売りに |
| G2: 主ラベル | `step_label` vs `step_label_dev` の AUC・решение性 | dev 単位の方が意味が通り性能も良い → 主指標を dev 単位へ（「モジュール別」は副次軸に） |
| G3: V(s) の価値 | `RL_iql_v_long_term.spearman ≥ 0.3` / `ensemble_best_beta` | 満たす → V(s) を最大の новизна として前面に。β≈1 (IRL単体で十分) → V は補助として書く |
| G4: 粒度 | event vs monthly の AUC・解釈性 | 計画どおり event 主軸を確認。逆転したら理由を考察 (§3 の分析で) |
| G5: 汎化 | `eval` vs `eval_devholdout` の AUC 差 | 差が大きい → 「既知 dev の記憶」批判に先回りして threats + dev 単位特徴の議論 |
| G6: RL の必要性 | `rf_supervised` / `logreg_supervised` との差 | RF に負ける軸があるなら、その軸では「補完性」(前論文の流儀) で書く |

γ 感度 (0.9/0.95/0.99) と done-mode ablation (`--done-mode dropout` で再ビルド→再学習)
は、主結果確定後に頑健性の節の材料として回す。

---

## Stage 3: 考察

### 3.1 推奨手法（実装済み）: V(s) 駆動要因分析

**RQ2「どのような状態遷移が離脱前兆か」への回答であり、この研究で一番
新規性のある考察**。`scripts/rl/analyze_value_drivers.py` が3点セットで出す:

```bash
uv run python scripts/rl/analyze_value_drivers.py \
    --dataset  outputs/rl/sweep/lstm_baseline_0-3m_event/dataset/event_transitions_test.parquet \
    --iql-model outputs/rl/sweep/lstm_baseline_0-3m_event/lam0.3_gam0.99/iql_agent.pt \
    --obs-dim 32 \
    --output-dir outputs/rl/analysis/event_lam0.3
```

| 出力 | 内容 | 論文での使い方 |
|------|------|---------------|
| `value_sensitivity.csv` + `fig_value_group_sensitivity.pdf` | 特徴グループ（負荷/承諾傾向/応答・間隔/活動量/協業/親和度）permutation 感度 | 「V は何を見ているか」の図。RQ2 の主図候補 |
| `dv_dx_spearman.csv` | 軌跡内の連続遷移で ΔV と Δ特徴の Spearman | 「拒否率上昇・応答遅延・活動間隔の増大が V 低下と共起」を定量で。ユーザーの中心仮説の直接検証 |
| `v_drop_profile.csv` | V 急落遷移（下位10%）での特徴変化の標準化プロファイル | 「離脱前兆の遷移で実際に何が起きているか」。考察の実例・ケーススタディの種 |
| `value_drivers_summary.md` | 上記の markdown まとめ | 考察節の下書きに貼る |

**書き方の注意**: これらは相関であり介入効果ではない。
「〜すると離脱する」ではなく「〜が V 低下と共起する / 離脱に先行する」と書く。

### 3.2 考察の方向性リスト（やれること全部）

推奨(3.1)で足りないとき・査読対策・結果次第で選ぶ。◎=推奨/低コスト、○=中、△=高コスト or 結果依存。

#### A. 価値関数 V(s) まわり

| # | 方向性 | 何が言えるか | コスト |
|---|--------|-------------|--------|
| A1 ◎ | 特徴グループ感度 + ΔV遷移分析 + 急落プロファイル | 上記 3.1 | **実装済み** |
| A2 ○ | V の分位で dev を層別 → 各層の実継続率・実残承諾数を集計（V の校正確認） | 「V は序列だけでなく水準としても意味がある」 | 小 (pair_predictions から集計するだけ) |
| A3 ○ | V 急落した具体的 dev のケーススタディ（時系列プロット 2〜3例） | 定性的な説得力。導入・考察の motivating example | 小 |
| A4 ○ | SHAP による個別状態の寄与分解 | A1 の個票版。図が映える | 中 (shap 依存追加) |
| A5 △ | V と負荷特徴の交互作用（高負荷 dev ほど重タスクで V が落ちるか） | 「負荷分散が効く相手」の特定 = RQ3 への橋 | 中 |
| A6 ○ | done-mode ablation（episode_end vs dropout）で V の Spearman 比較 | truncation バイアスの実証 → 手法の細部の正当化 | 小 (再ビルド+再学習1回) |

#### B. 方策・行動まわり

| # | 方向性 | 何が言えるか | コスト |
|---|--------|-------------|--------|
| B1 ○ | π_IQL と π_BC の乖離分析（advantage 加重が模倣を変えた状態の特徴付け） | 「IQL は単なる模倣を超えて何を学んだか」— RL の付加価値の直接証拠 | 小〜中 |
| B2 ○ | Q(s,1)−Q(s,0) の符号・分布分析（advantage<0 の状態 = 「今依頼すべきでない」状態のプロファイル） | actionable な知見。実務含意の節に | 小 |
| B3 △ | 行動3値化（承諾/明示的拒否/無応答）での再学習・再分析 | 「拒否」と「不在」の分離。RQ2 が鋭くなる | 中 (§2-1、response_time で判別) |

#### C. 原型・クラスタ（前論文と接続）

| # | 方向性 | 何が言えるか | コスト |
|---|--------|-------------|--------|
| C1 ○ | archetype 分析（前論文の手法）を V(s)・advantage に適用: 原型別に IQL vs IRL vs RF の AUC 差 | 前論文の「補完性」の物語を RL に拡張。継続研究としての一貫性 | 中 (既存 archetype コード再利用) |
| C2 △ | dev の V 軌跡（時系列）をクラスタリング → 離脱パターンの類型化 | 「急落型/漸減型/復活型」等の类型 = 実務で監視すべきパターン | 中 |

#### D. 時系列・遷移構造

| # | 方向性 | 何が言えるか | コスト |
|---|--------|-------------|--------|
| D1 △ | 離脱 dev の離脱前 k ヶ月の特徴軌跡を、継続 dev とマッチングして比較（準生存分析） | 「前兆は何ヶ月前から現れるか」— early warning の根拠 | 中〜大 |
| D2 △ | 遷移モチーフのパターンマイニング（例: 重タスク連続→拒否→無応答の系列頻度） | RQ2 を「系列パターン」として提示。図が強い | 大 |
| D3 ○ | リリースサイクル位相と V・承諾率の同期分析 | 前論文 6.1 の考察の定量化。プロジェクト状態特徴 (v2) の動機付け | 小〜中 |

#### E. 割り当て what-if（RQ3、スコープ判断後）

| # | 方向性 | 何が言えるか | コスト |
|---|--------|-------------|--------|
| E1 △ | 各依頼時点で候補 dev を Q/V スコアでランク → 実履歴・負荷ベース・承諾確率ベースとリプレイ比較（推定継続率 × 負荷 Gini） | 「V を使うと継続と負荷分散が両立する」— RQ3 の主結果 | 大 |
| E2 △ | FQE 等の off-policy evaluation で方策価値を推定 | E1 の主張を統計的に補強 | 大 |

#### F. 頑健性（threats への弾）

| # | 方向性 | コスト |
|---|--------|--------|
| F1 ◎ | λ・γ 感度（sweep 済み）、seed 分散、複数 t0 | 実行のみ |
| F2 ◎ | dev holdout（未知 dev 汎化、sweep 済み） | 実行のみ |
| F3 ○ | done-mode / buffer 期間 / event vs monthly の ablation 表 | 実行のみ〜小 |
| F4 △ | 第2エコシステム（Chromium/Android Gerrit）での再現 | 大 (データ取得) |

**推奨の組み合わせ（結果が想定どおりの場合）**:
主結果 = Stage 2 の AUC 表 + G3 の Spearman。考察 = A1（実装済み）を主軸に、
A2・B2 を安価に足し、前論文接続として C1。頑健性 = F1・F2。
RQ3 (E1) は8月末のスコープ判断まで保留。

---

## Stage 4: 論文執筆

### 4.1 RQ → 図表マッピング

| RQ | 主張 | 図表の素材 (生成元) |
|----|------|---------------------|
| RQ1: 逐次 RL は継続予測に有効か | IQL の π/V が IRL・RF・LogReg と同等以上、特に長期で | `summary.md` の AUC 表 + `methods_stats_*.csv` (CI/p値) + 多期間評価 (§0.5 の multitask 再抽出後) |
| RQ2: どの状態遷移が離脱前兆か | 負荷・拒否傾向・応答遅延の遷移が V 低下と共起 | `fig_value_group_sensitivity.pdf` + `dv_dx_spearman.csv` 上位表 + `v_drop_profile.csv` |
| RQ3 (任意): 割り当て what-if | Q/V ランキングで継続×負荷分散が改善（推定） | E1 のリプレイ表 (未実装) |
| 頑健性 | λ/γ/seed/t0/dev holdout に頑健 | `summary.csv` の pivot + devholdout 列 |

### 4.2 論文構成（SANER research track 想定、10+2p）

1. **Introduction** — turnover コスト → 予測から長期価値推定へ。motivating example に A3
2. **Background & Related Work** — 継続予測 (Zhou&Mockus, Bao, 前論文=匿名引用) /
   reviewer recommendation (workload-aware: Mirsaeedi&Rigby ICSE'20 と明確に差別化) /
   RL in SE (bug triage 等)
3. **Approach** — MDP 定義は rl_roadmap §3〜4 の時系列図をそのまま。
   IRL の位置づけは G1 の結果で確定
4. **Experimental Setup** — データ (OpenStack main32)、分割 (時間 + dev holdout)、
   ベースライン、指標。リーク防止の規律 (train-fit→test 固定適用) を明記
5. **Results (RQ1)** — AUC 表 + bootstrap CI。多期間 (1-4/4-6/7-9/10-12) で
   「長期ほど V が相対的に強い」が出れば主図に
6. **Analysis (RQ2)** — Stage 3 の3点セット
7. **Discussion** — 実務含意 (B2: いつ依頼を控えるべきか)、IRL との関係、RQ3 展望
8. **Threats** — 下のチェックリスト
9. **Conclusion**

### 4.3 Threats to Validity チェックリスト（先回りで書く）

- **Internal**: 行動は遷移・報酬に因果介入しない（advantage は相関。
  「介入効果」と書かない — 全節で表現チェック）/ ラベル参照期間の対称性 (§7.4) /
  train-fit 固定適用の規律 / sample_weight 等重みの選択
- **External**: OpenStack 単一エコシステム / 単一 t0 → rolling origin で緩和 /
  Gerrit ワークフロー依存
- **Construct**: 「継続」= directory 単位承諾の有無（dev 単位との比較を G2 で報告）/
  「非承諾」に無応答が混在 (B3 をやらない場合は明記) /
  done=データ打ち切りの扱い (F3 ablation)
- **Conclusion**: bootstrap CI・複数 seed・効果量。dev-month 重複による
  実効サンプル数の過大評価に言及

### 4.4 執筆時の注意（今までの議論の要点）

- double-anonymous: 前論文 (国内) は匿名引用処理。リポジトリ URL は伏せ、
  replication package は匿名化して Zenodo/Figshare 等へ
- 元論文のタスク再現はしない。「同じ土俵 (同一データ・同一情報・同一ターゲット)」
  で全手法比較、という立場を Setup で1段落使って明示 (§0.5 の整理)
- 「Bi-LSTM」誤記を持ち込まない (単方向2層 LSTM)
- γ・λ 等のハイパラは既定値と sweep 範囲を表で明示 (`train_config.json` が根拠)

---

## 付録: トラブルシュート

| 症状 | 原因と対処 |
|------|-----------|
| test transition が 0 件 | 抽出の train_end とリーク防止クリップの関係 (§7.5)。期間延長版の抽出スクリプトで軌跡を作り直す |
| parquet 書き出しで ImportError | pyarrow が環境に無い。`uv sync` (2026-07-04 に依存追加済み) |
| summary.md の表が出ない | tabulate 不足 → 同上 |
| devholdout eval が走らない | ビルド時に HOLDOUT_DEVS_RATIO 未設定。sweep は既定 0.1 で有効 |
| λ を変えても結果が同一 | train ログに「報酬を再計算」が出ているか確認 (`--reward-shaping` が渡っているか) |
| AUC が全手法 ≈0.5 | ラベル退化 (単一クラス) か、obs_dim 不一致 (state が切り詰められている)。ビルドログの state_dim / OBS_DIM を確認 |

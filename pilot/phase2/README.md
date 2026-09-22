# phase2 — パイロットの結論を国際学会水準まで詰める

phase1(`pilot/reports/loop*.md`)は「生成工程の AI 関与度が高いモジュールほど、評価工程の人間比率 H_E が下がる」
という相関を得た。phase2 はこれを (a) 分解し、(b) 因果として識別し、(c) メカニズムを特定するフェーズ。

**結論は `reports/FINDINGS_FULL.md`(全期間版、2023-01〜)を読むこと。**
`reports/FINDINGS.md` は13ヶ月版(2025-09〜2026-09)の記録で、現在は参考扱い。
`reports/PAPER_OUTLINE.md` に論文の骨子(タイトル案・Abstract・想定査読コメントへの反論)がある。

## 全期間データでの実行

入出力ディレクトリは環境変数で切り替える。既定値は13ヶ月版(phase1 と同じ)。

```
PILOT_RAW=raw_full PILOT_INTERIM=interim_full PILOT_PROCESSED=processed_full PHASE2_OUT=outputs_full PHASE2_FIG=figures_full uv run python pilot/phase2/scripts/d1_decomposition.py
```

収集から分析まで一括で回すなら `bash pilot/phase2/scripts/run_pipeline_full.sh`
(収集は `run_collect_full.sh`、失敗分の再試行は `retry_failed_collect.sh` が担当)。

phase1 の3つの結論を訂正した。詳細は FINDINGS.md §1, §5, §6。

1. H1b「評価工程からも人間が退く」→ **誤り**。比率低下は分母の膨張(dilution)で、人間の量は減っていない
2. RQ3「AI 関与度が高いと承認が +3.52日 遅れる」→ **撤回**。イベントスタディで再現しない(選択効果)
3. H2b'「参入障壁が下がり新規参入が増える」→ **撤回**。同上

## スクリプトの実行順序

入力は phase1 が作った `pilot/interim/*/events.parquet`, `pilot/raw/*/prs_full.jsonl` など。
出力はすべて `pilot/phase2/outputs/` に閉じる。

| # | スクリプト | 役割 |
|---|---|---|
| — | `common.py` | 全リポジトリの module x month 件数パネル構築、PanelOLS ヘルパー |
| D1 | `d1_decomposition.py` | H_E の分子・分母への分解(displacement か dilution か) |
| D2 | `d2_adoption_timing.py` | AI 参入タイミングの診断(イベントスタディが組めるか) |
| D3 | `d3_event_study.py` | モジュール単位の処置によるイベントスタディ → **失敗。記録として残す** |
| D4 | `d4_repo_event_study.py` | **リポジトリ単位の導入イベント。主たる因果識別** |
| D5 | `d5_pr_outcomes.py` | PR 単位アウトカムの構築(承認・マージ・レビュー種別)。D7/D10 の入力 |
| D6 | `d6_robustness.py` | 閾値感度 / stacked DiD / randomization inference |
| D7 | `d7_burden_event_study.py` | 人間の負担・PR の帰結 |
| D8 | `d8_talent_flow.py` | 新規参入・離脱 |
| D9 | `d9_equivalence.py` | 同値性検定(null result の裏付け) |
| D10 | `d10_review_kind.py` | **レビュー種別。メカニズムの核** |
| D11 | `d11_agent_profile.py` | エージェント別の行動プロファイル |
| D12 | `d12_figures.py` | 論文用の図(`figures/`) |
| D13 | `d13_review_response.py` | AI コメント後にコードが変わるか(レビュー単位 592,950件) |
| D14 | `d14_dose_response.py` | **用量反応**。AI を大量に入れても代替は起きないか |
| D15 | `d15_burden_distribution.py` | レビュー負荷の分配(集中度、コア/周辺) |
| D16 | `d16_generation_side.py` | 生成工程でも同じ構造か |
| D17 | `d17_asymmetry.py` | **生成と評価の非対称性の直接検定** → 非対称性は確認できず |
| D18 | `d18_redundancy.py` | AI レビューは冗長な重ね掛けか、空白の補完か |
| D19 | `d19_prediction_decay_full.py` | **SES研究との接続**。AI 参入前に学習した継続予測モデルは劣化するか(全期間データ必須) |
| — | `collect_full_history.py` / `run_collect_full.sh` / `retry_failed_collect.sh` / `run_pipeline_full.sh` | 2023-01〜の収集と、収集完了を待って全分析を回すパイプライン |

依存: D7/D10/D13/D18 は D5 の出力が必要。D6/D7/D8/D10/D14/D15/D18 は D4 の処置定義関数を、
D17 は D16 の処置定義関数を import する。
D3 は失敗した設計だが、「なぜモジュール単位ではだめか」を示すため残してある。
D16 単独では「生成側で人間が減る」ように見えるが、**D17 の直接検定で否定された**。D16 の結果だけを引用しないこと。

実行例: `uv run python pilot/phase2/scripts/d1_decomposition.py`

## 図

- `figures/fig1_dilution_trend.pdf` — 人間のレビュー量は増えているのに人間比率は下がる
- `figures/fig2_event_study.pdf` — 導入後、AI 側は増えるが人間側は動かない
- `figures/fig3_review_kind.pdf` — AI は COMMENTED しか出さない / 決定論的ボットは判断する

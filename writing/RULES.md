# 論文執筆ルール

# 厳守ルール
- 推測はせず**実装を確認して論文を記載すること**

## 表記ルール
- 「応諾」ではなく **「承諾」** を使う
- 「古典的」「古典的機械学習」は使わず **「従来の」** で揃える
- project（nova, cinder 等）と repo（nova/compute 等）を逆にしない
- 「ディレクトリ」は repo（depth=2 パス）を指す

### 英語語句の扱い
- 不要に英語を使わない。手法名・略語・特徴量名以外は日本語化する
- **保持する英語**: `MCE-IRL`, `IRL`, `RF`, `LSTM`, `Transformer`, `AUC`, `AUC-ROC`, `Random Forest`, `Archetypal Analysis`, `archetype`（単独）, `elbow`, `gain`, `bootstrap`, `Bonferroni`, `Wilcoxon`, 特徴量名（`path_review_count` 等）
- **archetype の名称は日本語のみ**（`A1 Sporadic Newcomer`, `P1 Unfamiliar Dir` 等の英語名は本文・表・図のいずれにも残さない）
- **置換ルール**:
  - `Uniform Ensemble` / `Ensemble` → **「アンサンブル」**（本研究では1種類しか提案しないため区別不要）
  - `dev archetype` → **「開発者archetype」**
  - `pair archetype` → **「ペアarchetype」**
  - `ペア単位pooled AUC` / `pooled AUC` → **「AUC」**（ペア単位ではpooledしか使わないので省略）
  - ただし開発者単位の `pooled dev-level AUC` は `per-dev AUC` と区別するため保持
- 「ペア」初出時は **「開発者-ディレクトリペア」**（ハイフン区切り）と明示、以降は「ペア」でよい
- 数式モード（`$...$`）内で日本語を使うときは `\mathrm{}` ではなく `\text{}` を使う（`\mathrm{}` は ASCII 専用）

## 引用スタイル
- 複数引用は `\cite{a}\cite{b}` の連結形式（`\cite{a, b}` 形式は使わない）
- 引用は **文末（句点直前）** に配置、文中には置かない

## 段落・組版
- 段落切替は **空行1行**（改行2個）で行う
- 段落先頭の全角スペース `　` は不要（`indentfirst` パッケージが自動で処理）
- `\todo{...}` マーカーは **残す**（作業中の印として保持）
- 文章を修正した箇所には必ず `\done{fix}` マーカーを末尾に挿入する（修正済みの印として保持）

## 節・図表参照
- 節・図表の参照には **`\ref{...}`** を使い，`$\S 5.2$` のような直書きはしない
- 節 (sub/subsubsection) には初出時 `\label{sec:xxx}` を付け，本文中では `\ref{sec:xxx}節` で参照する
- 図・表は既存通り `\label{fig:...}` / `\label{tab:...}` を用い，`\ref{fig:xxx}` / `\ref{tab:xxx}` で参照

## 避けたい表現
- 「陽に符号化する」 → 動作を素直に書く（例:「行動系列から報酬関数を推定する」）
- 「潰れる」 → 「失われてしまう」「区別できない」など
- 「原理的に困難である」 → 「捉えることが難しい」など
- 「診断する」「診断」「診断軸」「解釈軸」 → 「分析する」「分析手法」など
- 「窓」「予測窓」「短期窓」 → 「予測区間」（前向き）／「特徴量集計期間」（後ろ向き）
- 「out-of-time」 → 「訓練期間と評価期間を分離する構造」など日本語で書く
- 「短期 / 中期 / 中長期」のラベル → 具体的な月数で示す（例:「0--3か月」「3--9か月」）
- 過度に断定的な「効果がある」「有効である」は控えめに
- 「含意」「実運用への含意」 → 使わない（まとめ章でも使わない。内容を残す場合は「示唆」「知見」「得られた結果」等に置き換える）
- 「散発的」「散発的な〜」 → 使わない（「低活動」「不定期参加」等に置き換える）
- 「active」「active〜」 → 英単語のまま混入させない．条件文（例：「直前6か月にレビュー・PR履歴がある」）が並列にあるなら**冗長なので削除**，必要ならカタカナ「アクティブな〜」
- 「漏洩」 → 使わず **「リーク」** で統一（例：「ラベルリーク」「未来情報のリーク」）

## 図表
- 出力は **PDFのみ**（PNG は使わない）
- `savefig` 等では `.pdf` を指定（`format='pdf'`）
- **配置は `[H]` で本文直後に固定**（`float` パッケージが必要）。`[t]` だとフロートが意図しないセクションに流れ込むため使わない
- ページ下部の余白が大きくなりがちなときは preamble に `\raggedbottom` を入れて縦伸びを抑える
- 同種の情報を**表と図（ヒートマップ等）で重複させない**。数値表だけで足りる場合はヒートマップは不要
- 関連する複数の図を 1 つの figure 環境に **横並び**（`subcaption` + `minipage`）でまとめると紙面を節約できる

## 数式・パッケージ
- preamble に `amsmath`, `amssymb` を追加済み（`\mathbb{}`, `\text{}`, `cases` 環境などで必須）
- `placeins`（`\FloatBarrier`）, `float`（`[H]` 配置）, `subcaption`（横並び figure）も追加済み

## 結果セットとスコア源

### 使用モデル (この3種類のみ、他は一切使わない)
本研究で論文・分析・実装すべてに使うモデルは **次の3種類のみ**。これ以外の variant・列・出力は参照禁止。

| # | モデル | 確率列 (`pair_predictions.csv`) | 由来 |
|---|---|---|---|
| 1 | **cold IRL** | `irl_dir_prob` | MCE-IRL の素の出力（**キャリブレーション前**） |
| 2 | **RF** | `rf_dir_prob` | Random Forest の `predict_proba` |
| 3 | **アンサンブル** | `(irl_dir_prob + rf_dir_prob) / 2` | 上記2列の単純平均 |

### 使用禁止リスト (列・モデル variant)
以下は論文・分析スクリプトのいずれでも **参照禁止**。
- ❌ `irl_dir_prob_calibrated` (IsotonicRegression 事後校正後 — 使わない)
- ❌ `irl_global_prob` / `rf_global_prob` (global head — 使わない)
- ❌ warm-start IRL / kd_warmstart 系 (`main32_mce_kd_warmstart_*`)
- ❌ raw / profile などの他 variant ディレクトリ
- ❌ multitask 版・event IRL 版・教師あり LSTM 単独

### ディレクトリ
- **主結果ディレクトリは `outputs/main32_mce_volat_ep150/lstm_baseline/` に固定**（volat = volatility 特徴量入り、epoch=150）
- 4予測区間 (`train_0-3m/eval_0-3m`, `train_3-6m/eval_3-6m`, ...) の diagonal cell のみを評価対象とする
- Archetypal Analysis 用の alpha は `outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_dev_alpha_with_metrics.csv`（dev）と `archetype_pair_level_volat_ep150_coldonly/k4_archetype_profile.csv`（pair）

### 本文での扱い
- 上記方針に伴い，本文中で「キャリブレーション」「校正」「calibrated」「warm-start」「global head」に関する記述は導入しない
- 主指標は AUC-ROC（順位指標）であり，確率値そのものの校正は本研究のスコープ外
- 「アンサンブル」は単純平均1種類しか提案しないため，重み付き等の variant は議論しない

## ファイル構成
- 本文: `Hashimoto_SES.tex`
- bib: `Hashimoto.bib`
- 図: 配置場所は今後決定

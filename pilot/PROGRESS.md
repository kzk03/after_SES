# パイロット進捗管理

対象手順: `Claude outputs/パイロット手順.md`（`docs/研究設計_生成と評価の分離.md` §7 の実行版）
目的: 事前登録に書く内容を確定させること。論文用の結果を出すことではない。

このファイルは**作業ログ・状態管理**用。ゲートの正式な判定は `pilot/config/gates.md` に記録する（判定は人間が行う）。
運用: 上に新しいログを追記していく。古い記述は消さない。

---

## 対象リポジトリ

`home-assistant/core` と `grafana/grafana` の2件を、Loop 0〜6を通して並行で回す（パイロット手順の原則1を例外的に緩めた運用。詳細は下記ログ参照）。

## ステータス一覧

| # | 名前 | home-assistant/core | grafana/grafana | ゲート判定（gates.md参照） |
|---|---|---|---|---|
| 0 | 足場 | 完了(進む) | 完了(進む) | `config/gates.md` 記入済み |
| 1 | 担い手同定 | 完了(進む) | 完了(進む) | `config/gates.md` 記入済み。**agent定義=trailer込みに確定** |
| 2 | モジュール展開 | 完了(進む) | 完了(進む) | `config/gates.md` 記入済み |
| 3 | 工程の帰属 | 完了(設計変更) | 完了(設計変更) | `config/gates.md` 記入済み。**2工程(生成・評価)に縮小確定** |
| 4 | 指標の算出 | 完了(進む) | 完了(進む) | `config/gates.md` 記入済み。主指標はA_prime(trailer込み) |
| 5 | 分散の確認（★最重要） | 完了(進む) | 完了(進む) | `config/gates.md` 記入済み。**モジュール間比較で確定** |
| 6 | 分離度4軸 | 完了(進む) | 完了(進む) | `config/gates.md` 記入済み。S1(trailer込み)を主軸候補に |
| 7 | 拡大 | 完了(進む) | 完了(進む) | `config/gates.md` 記入済み。**6リポジトリ追加、主解析モデルの頑健性を確認** |
| 8 | 主軸決定 | 完了(下書き) | 完了(下書き) | `reports/loop8_preregistration.md` 作成済み。一部未定項目あり |

**Loop7で追加した6リポジトリ**: juspay/hyperswitch, ray-project/ray, python/cpython, ghostty-org/ghostty, apache/airflow, godotengine/godot(詳細は `reports/loop7_expansion.md`)

---

## 現在のブロッカー

（なし。GitHub token取得・対象リポジトリ決定済み）

---

## 次にやること

**ユーザーと相談し、Loop0〜6全ての正式ゲート判定を確定、`pilot/config/gates.md`に記入済み。** 決定事項:
- agent定義 = PR作成者login と commit trailer のOR条件(trailer込み)を主指標に採用
- 工程数 = 意図(I)を外し、生成(G)・評価(E)の2工程に設計変更
- 分離度の主軸 = S1(差分型、trailer込み版)を主軸候補に

**Loop8(事前登録の表)を先に作成完了。** `reports/loop8_preregistration.md` に主軸決定・比較設計・工程数・モジュール定義・感度分析リストをまとめた。未確定項目(有意水準、agent_identifiers.yaml公開版、リポジトリ最終リスト、k値、controls変数の実装)も明記済み。

次のアクション:
1. ~~Loop8の未確定項目のうち、今すぐ決められるもの~~ → 有意水準5%(両側検定)に確定(ユーザーと相談)
2. ~~主解析モデルの試運転~~ → **実施済み。A_mainの係数が有意に負(-0.0629, p<0.0001)でH1b(併進仮説)を支持する結果。** `reports/loop8_main_model_pilot.md` 参照。頑健性確認のためLoop7の優先度が上がった
3. Loop7(拡大)に着手するか判断。grafanaの検出力不足(iqr_diff=0.125)に加え、H1b支持という結果が2リポジトリだけの偶然でないか確認する意味でも優先度が高い
4. `docs/研究設計_生成と評価の分離.md` 本体を更新するか検討

## 重要な方針転換

- **対象期間を「2023-01〜直近」から「直近12ヶ月（2025-09〜2026-09）」に短縮。** 全期間だと1リポジトリ4〜5時間かかる見込みで、パイロット（ロジック検証・分散確認）の目的には過剰。issueは既に全期間分（2023-01〜）を収集済みでそのまま保持(軽量なため)。PRのみ期間短縮
- GraphQL API（`scripts/pilot/graphql_client.py` 等）でPR一括取得(files/reviews/commits込み、1クエリ100件でcost 3)に切り替え。REST APIでPRごとに叩く方式は非現実的な時間がかかるため不採用

---

## ログ

### 2026-09-12
- `Claude outputs/パイロット手順.md` と `docs/研究設計_生成と評価の分離.md` を確認し、全体像を把握
- `pilot/{config,raw,interim,processed,reports}/` の雛形を作成、`.gitignore` に `pilot/raw|interim|processed/` を追加（生データ・中間データは再取得/再生成可能なため。config/reportsはコミット対象）
- `pilot/config/gates.md` にゲート記録テンプレートを作成
- 本ファイル（進捗管理md）を新規作成
- Loop 0 着手前にブロッカー（GitHub token・対象リポジトリ）をユーザーに確認 → token は新規発行予定、対象リポジトリはこちらで候補調査する方針に決定
- **未認証での事前調査（Loop 0 の一部を先取り）**
  - ネットワーク到達性OK。未認証レート制限を実測: `core`(REST全般) 60回/時、`search` 10回/分、`code_search` 60回/時（`GET /rate_limit` で確認）
  - CODEOWNERS 保有確認（`raw.githubusercontent.com` 経由、非コメント・非空行のエントリ数）
    - `home-assistant/core`: 2191 行
    - `elastic/kibana`: 3383 行
    - `grafana/grafana`: 1286 行
    - `microsoft/vscode`: 12 行（ワークフロー/lint許可ファイルのみ。モジュール分割としては薄すぎる）
    - `langchain-ai/langchain`: 3 行、`vercel/next.js`: 0 行 → 実質なし、候補から除外
  - 直近PR 50件のauthorサンプル（`GET /repos/{repo}/pulls?state=all&per_page=50`）
    - `microsoft/vscode`: `Copilot` ログインのPRが3件（GitHub Copilot coding agent と思われる）
    - `grafana/grafana`: `cursor[bot]` が1件
    - `elastic/kibana` / `home-assistant/core`: renovate/dependabot系のみで、コーディングエージェントらしきものは未検出（n=50のサンプルなので断定不可）
  - 既知AIエージェントbot（`copilot-swe-agent` / `devin-ai-integration` / `cursor` / `claude` / `codegen-sh` / `sweep-ai`）の出現件数を `search/issues` で確認しようとしたが、**未認証の search API レート制限（10回/分）に到達し中断**。token取得後に正式実行が必要
- **暫定候補**: CODEOWNERSのモジュール数が十分（1000+行）な `home-assistant/core` / `grafana/grafana` / `elastic/kibana` の3つに絞り込み。最終1件はtoken取得後、AIエージェントPR件数を正式比較して決定する
- **GitHub token受領・認証確認**: `.env` に `GITHUB_TOKEN` 追加を確認。認証済みレート制限を実測 → `core` 5000/時、`search` 30/分、`graphql` 5000/時（未認証は `core` 60/時, `search` 10/分）
- **対象リポジトリ最終決定の調査**
  - `search/issues?q=repo:{repo}+is:pr+author:app/{bot}` で正式カウント: `home-assistant/core` copilot-swe-agent **34件**／`grafana/grafana` cursor **87件**／`elastic/kibana` claude **2件**、他のbotはほぼ0件
  - `elastic/kibana` はAIエージェントPRが少なすぎるため除外
  - PRタイトルを目視サンプル: `home-assistant/core` のCopilot PRは「小規模バグ修正・機能追加の自律生成」で一貫。`grafana/grafana` の cursor[bot] は eslint suppression修正（Agent風）／`[fep-auto-deps]` 依存bump（legacy_bot風）／`docs sync [automated]`（スクリプト風）が同一アカウントに混在しており、Loop1の担い手分類でノイズになるリスクあり
- **方針決定（ユーザー判断）**: パイロット手順の原則1（1リポジトリで通してから広げる）を今回は例外的に緩め、**`home-assistant/core` と `grafana/grafana` の2リポジトリをLoop 0〜6を通して並行で回す**。`gates.md` には両リポジトリ分の判定を記録する
- **Loop 0 実行**: `scripts/pilot/github_client.py`（薄いAPIラッパー）と `scripts/pilot/loop0_scaffold.py`（試験取得本体）を実装し、両リポジトリで issue/PR/review/commit/CODEOWNERS を試験取得（`pilot/raw/<owner>__<repo>/*.jsonl`）
  - review は両リポジトリで取得成功（home-assistant: 20PR中20PRにreview計65件／grafana: 20PR中5PRにreview計8件）
  - **発見**: review提出者の大半が自動化アカウント。home-assistantは65件中37件が `copilot-pull-request-reviewer[bot]`（GitHub Copilotの自動レビュー機能）。これをagent/legacy_botのどちらに分類するかがLoop1の重要な検討事項
  - CODEOWNERS履歴も両リポジトリで取得成功（home-assistant 1970コミット/2017〜2026年、grafana 937コミット/2019〜2026年）。home-assistantのCODEOWNERS変更コミットに `Co-authored-by: Copilot` trailerを実例確認
  - 詳細は `pilot/reports/loop0_scaffold.md` 参照。ゲート判定は人間が行うため `pilot/config/gates.md` は未記入
- **ユーザー指示「一旦パイロット回し切って」を受け、Loop 1〜6 を通しで実装・実行**（ゲートで都度停止せず、判定材料をまとめてから提示する方針）
  - GraphQL API(`scripts/pilot/graphql_client.py`, `collect_prs.py`, `collect_issues.py`)でPR/issue全件を収集。対象期間は「全期間だと1リポジトリ4〜5時間かかる」ため直近12ヶ月に短縮(issueは全期間分を既存収集分のまま流用)
  - PR収集中にGitHub API側の502/504エラーとネットワーク瞬断で計4回中断。リトライ処理を強化(接続エラー・不完全レスポンスもリトライ対象に、リトライ回数15回に増加)して再開を繰り返し完走。home-assistant 19091件、grafana 17496件のPRを取得
  - **Loop1(担い手同定)**: `pilot/config/agent_identifiers.yaml`をMSR2026 "AIDev"データセット対象6エージェント(Copilot/Cursor/Devin/Claude Code/Google Jules/OpenAI Codex)の実App識別子で作成。actor login分類は目視確認で誤分類なし
  - **★重要発見**: actor loginだけで見るとagentは6〜7アカウントのみだが、commitのtrailer(`Co-authored-by: Claude`等)では数千件検出。AIエージェントは「エージェント自身のアカウントが自律PR作成」より「人間アカウントの下で使われCo-authored-byが残る」形が主流と判明
  - **バグ修正**: Loop4でtrailer込みのH_G_prime/A_prime計算時、`~series.fillna(False)`がobject dtypeでビット反転(真偽反転にならない)になるバグを発見・修正(`astype(bool)`を追加)。修正前はA_prime=Aと誤って一致していた
  - **Loop5(★最重要ゲート)の結果がactor login版とtrailer込み版で真逆に近い**: login版は両リポジトリとも「分散なし、断念」相当(iqr_diff=0.0)。trailer込み版はhome-assistant「設計通り進む」(iqr_diff=0.25)、grafana「進むが不安」(iqr_diff=0.125)。**agentの定義をどちらにするかが研究のGo/No-Go判断そのものになっている**
  - **Loop3**: issueのモジュール帰属率が7.3%/8.4%と大幅に基準(50%)未達、30%未満のため「2工程に縮小」ゲートに該当。跨るPRも52%超で主モジュール方式の妥当性に疑義
  - **Loop6**: S1_EG(評価-生成の人間比率差)が両リポジトリでほぼ負。「評価に人間が残る」という講演の主張と逆方向の可能性。4軸の相互相関は低め(0.7未満)
  - 詳細は `pilot/reports/loop{1,3,4,5,6}_*.md` 参照。全て判定欄は空欄(ユーザー判断待ち)
- **「loop再開してほしい」を受け、持ち越し論点のうち実装可能な分を追加実施**
  - Loop6 S4(時間的先行型)をH_G_prime(trailer込み)でも再計算。**login版は計算できたモジュールが7件/15件(事実上「計算不能」)だったが、trailer込みでは336件/119件に激増**。ラグ0(同時性)が最頻値。「S4計算不能」という判定もagent定義次第で覆ることが判明
  - Loop6 S2(Jaccard)のtrailer込み拡張は見送り: G側にtrailer検出エージェントを追加してもE側の実在アカウント名と一致せず分子に寄与しない(名寄せ表が必要、Loop8までの持ち越しとした)
  - Loop3の「3工程→2工程縮小」の影響を試算: 2工程化で失われるパネル行はわずか6%程度、かつ研究設計書§5.1の主解析(`H_E~A`)にはH_Iが登場しないため**主解析への影響はゼロ**。失われるのはH1c(補助仮説)のみ。**2工程への縮小を推奨**の所見をレポートに追記
  - Loop7(拡大)は着手せず: agent定義(login/trailer)が確定しないまま数十リポジトリに手を広げるとAPI消費・時間コストに対して手戻りリスクが大きいため保留
- **ユーザーと相談し、最重要論点を確定**(`pilot/config/gates.md` に正式記入、`pilot/config/agent_identifiers.yaml` にも反映)
  - agent定義: **PR作成者login と commit trailer のOR条件(trailer込み)を主指標に採用**。login単独は感度分析として残す
  - 工程数: **意図(I)を外し、生成(G)・評価(E)の2工程に設計変更**。理由は主解析(`H_E~A`)にH_Iが登場せず、2工程化で失うパネル行はわずか6%程度のため
  - Loop5(★最重要ゲート)の判定が確定: **モジュール間比較を主軸に進める**(home-assistantは基準クリア、grafanaは検出力に不安ありだが2リポジトリ合わせて成立と判断)
  - Loop0〜6全て `config/gates.md` に正式判定を記入完了
- **Loop8(事前登録)着手**: `reports/loop8_preregistration.md` を作成。主軸・比較設計・工程数・モジュール定義・感度分析リストを確定し `config/gates.md` にもLoop8エントリを追記
  - 判定基準の有意水準を**5%(両側検定)に確定**(ユーザーと相談)
  - 残る未定項目: agent_identifiers.yaml公開版(MSR論文精読待ち)、リポジトリ選定手順(Loop7待ち)、k値の主値(Stage2)、controls変数の実装
- **「AI研究の方は自ら判断して進めてほしい」との指示を受け、主解析モデルを試運転**(`scripts/pilot/loop8_main_model.py`、`statsmodels`/`linearmodels`を追加)
  - `H_E(m,t) ~ A_prime(m,t) + log(1+n_files) + log(1+n_events) + log(1+n_unique_actors) + module_FE + time_FE` をPanelOLS(モジュール単位クラスタ標準誤差)で推定
  - **★A_mainの係数 = -0.0629、p<0.0001で有意に負。gates.md Loop8の判定基準に従うと「H1b(併進仮説)支持」= 講演の主張(評価には人間が残る)は支持されない**という結果
  - log_n_files(モジュール規模)はモジュール固定効果に完全吸収され推定不能(時間不変変数のため設計見直しが必要)
  - 2リポジトリのみでリポジトリ単位クラスタ標準誤差が実行不能なため、この結果はあくまでパイロット試運転。Loop7で再現するか確認が必要
  - 詳細は `pilot/reports/loop8_main_model_pilot.md` 参照
- **Loop7(拡大)を実施し、H1b支持という結果の頑健性を確認**(`reports/loop7_expansion.md`)
  - GitHub Search APIで人気リポジトリ1000件 → CODEOWNERS保有271件 → エントリ数フィルタ32件 → 2023-01以前作成に絞りA(trailer込み軽量推定)で高中低3層から2件ずつ抽出: juspay/hyperswitch, ray-project/ray, python/cpython, ghostty-org/ghostty, apache/airflow, godotengine/godot
  - 6リポジトリ全てでLoop1〜4パイプライン完走(issueは2工程決定により収集省略)
  - **★主解析モデルを8リポジトリ(既存2+新規6)で再実行 → A_mainの係数 -0.0629→-0.0624とほぼ不変。初めてリポジトリ単位クラスタ標準誤差(設計書指定の本来の方法)でも計算でき、p=1.41e-9で高度に有意。H1b支持という結論が頑健に再現**
  - 技術的な学び: GraphQL APIの6プロセス同時並行実行はGitHubのセカンダリレート制限(403)に抵触した。起動間隔をずらし、403時に長め待機するようリトライ処理を強化して対応
  - 層あたりサンプル数(2件)は手順書の目安(10件以上)に未達。正式なLoop7本格実施は今後の判断
- **controls変数(モジュール規模)の設計見直し**: log_n_filesが時間不変でモジュール固定効果に吸収される問題を修正。メインモデルから除外し、between推定(モジュール単位平均、repo固定効果込みpooled OLS)を追加 → **between推定でもA_mainは-0.0612, p<0.0001で有意に負。within推定(-0.0624)と一致し、H1b支持の頑健性がさらに高まった**
- **正式Loop7を実施(層あたり10件以上に拡大)**: GitHub Search APIで候補母集団を2200件超に拡大し、31リポジトリを追加投入(node.jsはGraphQL内部エラーが解消せず除外)。**既存2+第1弾6+追加31=計39リポジトリ**で主解析モデルを再実行
  - **A_mainの係数: -0.0629(2リポ)→-0.0624(8リポ)→-0.0530(39リポ)。n_obs=30186, n_repo_clusters=39でp=0.0044(リポジトリクラスタ)、p=2.86e-12(モジュールクラスタ)。H1b支持が大規模・多様なサンプルでも頑健に再現**
  - 技術的な課題に多数遭遇し対応: xargs -Pの環境依存エラー(forループ+&+waitに変更)、`files`フィールドがnullになるケース(`.get() or {}`でガード追加)、node.jsでGitHub側のGraphQL INTERNALエラーが解消せず除外
- **RQ2(離脱率・新規参入)を実測**(`scripts/pilot/rq2_outcomes.py`, `rq2_model.py`。39リポジトリ全部でLoop6未実施だったため先に実行してから着手)
  - **H2a(離脱率上昇)は不支持**: S1_EG_primeとchurn_rateの間に固定効果統制後の有意な関係なし(単純相関では弱い負の相関があったが消えた)
  - **H2b(新規参入減少)は逆方向の結果**: 分離度が低いモジュール(評価もAI化)ほど新規参入が多い(k=6でp=0.010〜0.012、repoクラスタでも有意)。「参入障壁低下」「ホットスポット効果の交絡」等複数の解釈が可能で未確定
  - 詳細は `pilot/reports/rq2_outcomes.md` 参照

# Loop 0 — 足場

対象: `home-assistant/core`、`grafana/grafana`（2リポジトリ並行）
実行スクリプト: `scripts/pilot/loop0_scaffold.py`（薄いAPIラッパー: `scripts/pilot/github_client.py`）
実施日: 2026-09-12

---

## 1. やったこと

1. GitHub REST API の認証・レート制限を実測
2. 両リポジトリで issue / PR / review / commit / CODEOWNERS を試験取得し、`pilot/raw/<owner>__<repo>/` に jsonl で保存
   - issues: 直近100件（`_is_pr` フラグでPRと区別）
   - PRs: 直近100件
   - reviews: 直近PR 20件分の review を全件取得
   - commits: 直近100件
   - CODEOWNERS: パスを指定して全コミット履歴を取得（月次スナップショット化はLoop 2で本実装）
3. 各レコードに `_fetched_at`（取得日時, UTC ISO8601）を付与

## 2. レート制限の実測

| 状態 | core | search | graphql |
|---|---|---|---|
| 未認証 | 60/時 | 10/分 | 0（利用不可） |
| 認証済み（fine-grained PAT, read-only） | 5000/時 | 30/分 | 5000/時 |

> 実行前後で `/rate_limit` の `used` が 0 のままだった。このエンドポイント自体は消費にカウントされない可能性が高く、正確な消費数は取れていない（概算では issues 1回 + PRs 1回 + reviews 20回 + commits 1回 + CODEOWNERS履歴 約20〜30回 + content 1回 ×2repo ≒ 90〜110回程度）。Loop 2以降で本格収集する際は `X-RateLimit-Remaining` ヘッダーを都度ログすること。

## 3. 最優先確認事項の結果

### review イベントは取れるか

**取れる。** 両リポジトリで実際に提出された review（`GET /repos/{repo}/pulls/{n}/reviews`）を取得できた。

| リポジトリ | チェックしたPR数 | reviewが付いていたPR数 | 取得review総数 |
|---|---|---|---|
| home-assistant/core | 20 | 20 | 65 |
| grafana/grafana | 20 | 5 | 8 |

**⚠ 発見（Loop 1 に引き継ぐべき重要事項）**
review の提出者内訳を見ると、**自動化アカウントが大半を占める**。

- `home-assistant/core`（65件）: `copilot-pull-request-reviewer[bot]` **37件**、`home-assistant[bot]` 7件、人間 21件
- `grafana/grafana`（8件、サンプル数が少なく参考値）: `cursor[bot]` 3件、`github-actions[bot]` 2件、`grafana-pr-approver[bot]` 1件、人間 2件

`copilot-pull-request-reviewer[bot]` は GitHub Copilot の自動コードレビュー機能であり、研究設計書 §3.1 の分類でいう **`agent`**（従来型bot＝`legacy_bot` ではなく）に該当する可能性が高い。これが事実なら、**「評価工程に人間が残る」という分離仮説(H1a)にとって不利な材料**になり得る（評価もagentが担っている割合が無視できない）。Loop 1 の担い手同定で `copilot-pull-request-reviewer[bot]` 等の**自動レビューbotをagent/legacy_botのどちらに分類するか**を明確な基準として決める必要がある。

### CODEOWNERS の履歴は取れるか

**取れる。**

| リポジトリ | パス | コミット数 | 最古 | 最新 |
|---|---|---|---|---|
| home-assistant/core | `CODEOWNERS` | 1970 | 2017-07-07 | 2026-09-11 |
| grafana/grafana | `.github/CODEOWNERS` | 937 | 2019-12-11 | 2026-09-11 |

現在時点の内容取得（`GET /repos/{repo}/contents/{path}`）も両リポジトリで成功。月次スナップショットの構築（各月の該当時点のコミットを選んで内容を取得）はLoop 2で実装する。

**副次的な発見**: home-assistant/core の CODEOWNERS 変更コミットの中に `Co-authored-by: Copilot <...>` trailer を含むものを確認した。担い手同定（Loop 1）での trailer ベースのagent識別が機能する実例。

## 4. ゲート判定（案。判定は人間が行う）

パイロット手順.md のゲート表:

| 条件 | 判断 |
|---|---|
| review が取れる | → Loop 1 へ |
| review は取れるが requested_reviewers が取れない | → 問題なし |
| review も取れない | → 設計を根本から見直す |

**両リポジトリとも「review が取れる」に該当**するため、条件上は Loop 1 へ進める状態。

ただし上記の発見（自動レビューbotの比率が高い）を踏まえ、Loop 1 の設計に以下を持ち越す:

- [ ] `copilot-pull-request-reviewer[bot]` 等、GitHub純正の自動レビュー機能をどう分類するか（`agent` / `legacy_bot` / 主解析から別途除外、の3択を検討）
- [ ] grafana/grafana は review 密度自体が home-assistant/core より低く見える（20PR中5PRのみ）。サンプル数を増やして実際の分布を確認する必要あり

`pilot/config/gates.md` への正式な記入（判定・決めたこと・持ち越した未決）はユーザー判断を待つ。

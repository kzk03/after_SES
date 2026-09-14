# Loop 1 — 担い手同定

対象: `home-assistant/core`、`grafana/grafana`(直近12ヶ月、2025-09〜2026-09)
実行スクリプト: `scripts/pilot/loop1_actors.py`
設定: `pilot/config/agent_identifiers.yaml`(パイロット版v0。MSR 2026 Mining Challengeの"AIDev"データセットが対象とする6エージェントの公式App識別子ベース)
実施日: 2026-09-12

## やったこと

1. PR作成者・review提出者・commit作成者(login)を全収集し、agent_identifiers.yamlで human/agent/legacy_bot に分類
2. commitメッセージに対しtrailerパターン(`Co-authored-by: Claude` 等)をスキャンし、出現件数を集計
3. 各カテゴリから最大50件を無作為抽出し目視確認(`loop1_visual_check_sample.csv`)

## 結果: actor login ベースの分類

| リポジトリ | 全actor数 | human | agent | legacy_bot |
|---|---|---|---|---|
| home-assistant/core | 24847 | 24829 | 7 | 11 |
| grafana/grafana | 6321 | 6300 | 6 | 15 |

目視確認したサンプル(各リポジトリ100件強)では、**誤分類は確認されなかった**。legacy_bot判定はaccountsリストだけでなく `.*\[bot\]$` パターンのフォールバックが効いており、リスト未登録のプロジェクト固有bot(`grafana-delivery-bot[bot]` 等)も正しく捕捉できていた。

## ★重要な発見: commit trailer による検出が本質的に重要

trailerパターンのヒット件数(コミット単位):

| パターン | home-assistant/core | grafana/grafana |
|---|---|---|
| `Co-authored-by:.*claude` | 4258 | 3964 |
| `Co-authored-by: Copilot` | 2144 | 93 |
| `Co-authored-by:.*cursor` | 184 | 651 |
| `Co-authored-by:.*copilot-swe-agent` | 8 | 0 |

**actor login(PR作成者アカウント自体)ベースの`agent`はわずか6〜7アカウントしかないのに対し、trailerベースでは数千件のコミットでAIエージェント関与が検出された。** これは、home-assistant/core・grafana/grafanaのいずれも、**AIエージェントの主な使われ方は「人間がPRの作成者のまま、Claude CodeやGitHub Copilotを使って実装し、コミットにCo-authored-byを残す」形**であり、「エージェント自身のアカウントが自律的にPRを作る」形(GitHub Copilot coding agent, Cursor Agent等)はまだ少数派であることを意味する。

研究設計書§3.1は「commit trailerに既知のエージェント識別子を含むものもagentとする」と規定しているが、当初のLoop1実装(actor loginのみで分類)はこの規定を反映できていなかった。Loop3/Loop4でtrailer検出を反映した`H_G_prime`/`A_prime`を追加実装したところ、Loop5(分散確認)の結果が大きく変わった(詳細は`loop5_variance.md`)。

## ゲート判定(案。判定は人間が行う)

| 指標 | 基準 | 結果 |
|---|---|---|
| strict分類の精度(目視) | ≥95% | サンプル内で誤分類なし。基準を満たすと見られる |
| agentと判定されたactor数 | ≥1 | 満たす(login単位で6〜7、trailer単位では数千件) |
| legacy_botをagentと誤判定 | 0件 | 誤判定なし |

**条件上は「進む」に該当するが、"何をもってagent関与とするか"(login単位かtrailer単位か)によって後続Loopの結論が変わるため、この判断はLoop8(主軸決定)まで持ち越さず、ここで確定させるべき論点として明記する。**

## 持ち越した論点

- [ ] **agent判定の主軸をtrailerベースにするか、actor loginベースにするか(またはOR条件で統合するか)を確定する。** 現状はLoop4/5でA(login)とA_prime(login+trailer)を並記しているが、事前登録時にはどちらか一方を主指標にする必要がある
- [ ] MSR 2026 Mining Challenge採録論文10本の精読はまだ完了していない(時間の都合で "AIDev" データセットの対象6エージェントで代用)。事前登録前に照合する
- [ ] `Copilot`(PR作成者としての表示名)と`copilot-swe-agent`(login)が同一エンティティの可能性がある。名寄せの要否を確認

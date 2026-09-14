"""GitHub GraphQL API の薄いラッパー。ページネーション・チェックポイント再開・レート制限待機を扱う。"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

GRAPHQL_URL = "https://api.github.com/graphql"


class GraphQLClient:
    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN が未設定です(.env を確認してください)")
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        )

    def query(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(15):
            try:
                resp = self.session.post(GRAPHQL_URL, json={"query": query, "variables": variables}, timeout=90)
            except requests.exceptions.RequestException as e:
                last_err = e
                wait_s = 5 * (attempt + 1)
                print(f"  [retry] connection error: {e}, {wait_s}s待機")
                time.sleep(wait_s)
                continue
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError as e:
                    # サーバー側でレスポンスが途中で切れることがある(重いクエリ時の504寄りの挙動)。リトライ対象にする
                    last_err = e
                    wait_s = 5 * (attempt + 1)
                    print(f"  [retry] 不完全なレスポンス({e}), {wait_s}s待機")
                    time.sleep(wait_s)
                    continue
                if "errors" in data:
                    errors = data["errors"]
                    # NOT_FOUND(削除済みユーザー等でnullableフィールドが解決できないだけ)は
                    # 該当フィールドがnullになるだけで致命的ではないので、dataがあれば続行する
                    non_fatal_types = {"NOT_FOUND"}
                    retryable_types = {"INTERNAL"}  # GitHub側の一時的な内部エラー(特定ノードでクエリが重すぎる場合など)
                    fatal = [e for e in errors if e.get("type") not in non_fatal_types | retryable_types]
                    if fatal:
                        raise RuntimeError(f"GraphQL errors: {errors}")
                    if any(e.get("type") in retryable_types for e in errors) or not data.get("data"):
                        wait_s = 5 * (attempt + 1)
                        print(f"  [retry] INTERNAL error ({errors}), {wait_s}s待機")
                        time.sleep(wait_s)
                        continue
                    print(f"  [warn] 部分的なNOT_FOUNDエラー({len(errors)}件, 該当フィールドはnull扱いで続行)")
                rl = data.get("data", {}).get("rateLimit")
                if rl and rl["remaining"] < 50:
                    wait_s = 60
                    print(f"  [rate-limit] remaining={rl['remaining']}, {wait_s}s待機")
                    time.sleep(wait_s)
                return data["data"]
            if resp.status_code in (502, 503, 504):
                wait_s = 5 * (attempt + 1)
                print(f"  [retry] status={resp.status_code}, {wait_s}s待機")
                time.sleep(wait_s)
                continue
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                # secondary rate limit: 複数プロセス並行実行時に発生しやすい。長めに待つ
                retry_after = resp.headers.get("Retry-After")
                wait_s = int(retry_after) if retry_after else 90 * (attempt + 1)
                print(f"  [retry] secondary rate limit, {wait_s}s待機")
                time.sleep(wait_s)
                continue
            raise RuntimeError(f"GraphQL POST -> {resp.status_code}: {resp.text[:500]}")
        raise RuntimeError(f"GraphQL リトライ上限に達しました(最後のエラー: {last_err})")


def paginate_connection(
    client: GraphQLClient,
    query: str,
    variables: dict[str, Any],
    connection_path: list[str],
    checkpoint_path: Path,
    out_path: Path,
    stop_condition: Callable[[dict[str, Any]], bool] | None = None,
) -> int:
    """connection_path で指定した nodes/pageInfo を持つ接続をページネーションしながら jsonl に追記する。

    checkpoint_path に {"cursor": ..., "count": ...} を保存し、中断後に再開できるようにする。
    stop_condition(node) が True を返したら、それ以降の取得を打ち切る(古い順ページングで期間外に出た場合など)。
    """
    cursor = None
    count = 0
    if checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text())
        cursor = ckpt.get("cursor")
        count = ckpt.get("count", 0)
        print(f"  再開: cursor={cursor}, 既存件数={count}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_path.exists() and cursor else "w"
    stopped_early = False

    with out_path.open(mode, encoding="utf-8") as f:
        while True:
            vars_ = dict(variables)
            vars_["cursor"] = cursor
            data = client.query(query, vars_)
            conn = data
            for key in connection_path:
                conn = conn[key]
            nodes = conn["nodes"]
            for node in nodes:
                if stop_condition is not None and stop_condition(node):
                    stopped_early = True
                    break
                f.write(json.dumps(node, ensure_ascii=False) + "\n")
                count += 1
            if stopped_early:
                break
            page_info = conn["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            cursor = page_info["endCursor"]
            checkpoint_path.write_text(json.dumps({"cursor": cursor, "count": count}))
            if count % 1000 < 50:
                print(f"  ...{count}件取得済み")

    checkpoint_path.write_text(json.dumps({"cursor": None, "count": count, "done": True}))
    return count

"""GitHub REST API の薄いラッパー。

パイロット手順.md Loop 0 用。認証・ページネーション・レート制限待機のみを扱う。
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

API_BASE = "https://api.github.com"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GitHubClient:
    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN が未設定です(.env を確認してください)")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def _wait_for_rate_limit(self, resp: requests.Response) -> None:
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        if remaining is not None and int(remaining) == 0 and reset is not None:
            wait_s = max(0, int(reset) - int(time.time())) + 1
            print(f"  [rate-limit] remaining=0, {wait_s}s待機します")
            time.sleep(wait_s)

    def get(self, path: str, params: dict[str, Any] | None = None, max_retries: int = 5) -> requests.Response:
        url = path if path.startswith("http") else f"{API_BASE}{path}"
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                wait_s = 5 * (attempt + 1)
                print(f"  [retry] connection error: {e}, {wait_s}s待機")
                time.sleep(wait_s)
                continue
            if resp.status_code in (502, 503, 504):
                wait_s = 5 * (attempt + 1)
                print(f"  [retry] status={resp.status_code}, {wait_s}s待機")
                time.sleep(wait_s)
                continue
            self._wait_for_rate_limit(resp)
            return resp
        raise RuntimeError(f"GET {url} リトライ上限に達しました")

    def paginate(
        self, path: str, params: dict[str, Any] | None = None, max_pages: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """GET を続けてページネーションし、items を1件ずつ yield する。"""
        params = dict(params or {})
        params.setdefault("per_page", 100)
        page = 1
        while True:
            p = dict(params)
            p["page"] = page
            resp = self.get(path, params=p)
            if resp.status_code != 200:
                raise RuntimeError(f"GET {path} page={page} -> {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            if not data:
                break
            for item in data:
                yield item
            if len(data) < params["per_page"]:
                break
            page += 1
            if max_pages is not None and page > max_pages:
                break

    def rate_limit(self) -> dict[str, Any]:
        resp = self.get("/rate_limit")
        return resp.json()

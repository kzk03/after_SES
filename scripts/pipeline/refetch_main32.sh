#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# [SES 再収集] main32（論文スコープ32リポジトリ）を Gerrit から再取得する。
# ──────────────────────────────────────────────────────────────────────────
# collect_service_teams.sh は 244 repo 版 CSV（'repo' 列）用。main32 は
# service_teams_main_repos.csv の 'main_repo' 列なので、ここでは main_repo を
# 直接読んで build_dataset.py を 1 repo ずつ呼ぶ。
#
# 既存データ（data/raw_json, data/combined_raw_main32.csv）は触らず、別ディレクトリへ保存。
# 再収集すると当時とデータが変わる（abandoned change 消失等）ため、原本保護が目的。
#
# 使い方:
#   bash scripts/pipeline/refetch_main32.sh [out_base] [parallel]
# 例:
#   bash scripts/pipeline/refetch_main32.sh data/refetch_20260615 4
#
# 完了後: build_dataset 済みの per-repo CSV を結合して combined を作る手順は
#   末尾の [次の手順] を参照。
# ──────────────────────────────────────────────────────────────────────────
set -u

OUT_BASE="${1:-data/refetch_20260615}"
PARALLEL="${2:-4}"
GERRIT_URL="${GERRIT_URL:-https://review.opendev.org}"
START_DATE="${START_DATE:-2020-01-01}"
END_DATE="${END_DATE:-2026-01-01}"
RESPONSE_WINDOW="${RESPONSE_WINDOW:-14}"
MAIN_CSV="${MAIN_CSV:-data/service_teams_main_repos.csv}"
SLEEP_PER_REPO="${SLEEP_PER_REPO:-5}"

RAW_JSON_DIR="$OUT_BASE/raw_json"
RAW_CSV_DIR="$OUT_BASE/raw_csv"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$RAW_JSON_DIR" "$RAW_CSV_DIR" "$LOG_DIR"

echo "============================================================"
echo "  [main32 再収集] OUT_BASE=$OUT_BASE PARALLEL=$PARALLEL"
echo "  期間 $START_DATE 〜 $END_DATE / response_window=$RESPONSE_WINDOW"
echo "  保存先 $RAW_JSON_DIR （原本 data/raw_json は非破壊）"
echo "============================================================"

# main_repo 列（"openstack/adjutant" 等）を取得。sunbeam は除外。
# （macOS の bash 3.2 には mapfile が無いので read ループで読む）
REPOS=()
while IFS= read -r line; do
    [ -n "$line" ] && REPOS+=("$line")
done < <(python3 -c "
import csv
with open('$MAIN_CSV') as f:
    for r in csv.DictReader(f):
        repo = r.get('main_repo')
        if not repo or r.get('team') == 'sunbeam':
            continue
        print(repo)
")
echo "対象 repo: ${#REPOS[@]} 件"

fetch_one() {
    local repo="$1"
    local safe="${repo//\//__}"
    local json_path="$RAW_JSON_DIR/${safe}.json"
    local csv_path="$RAW_CSV_DIR/${safe}.csv"
    if [ -f "$json_path" ] && [ -f "$csv_path" ]; then
        echo "[SKIP] $repo"
        return 0
    fi
    echo "[START] $repo"
    uv run python scripts/pipeline/build_dataset.py \
        --gerrit-url "$GERRIT_URL" --project "$repo" \
        --start-date "$START_DATE" --end-date "$END_DATE" \
        --response-window "$RESPONSE_WINDOW" \
        --output "$csv_path" --raw-output "$json_path" \
        > "$LOG_DIR/${safe}.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && echo "[DONE] $repo" || echo "[FAIL] $repo (rc=$rc, see $LOG_DIR/${safe}.log)"
    sleep "$SLEEP_PER_REPO"
    return $rc
}
export -f fetch_one
export RAW_JSON_DIR RAW_CSV_DIR LOG_DIR GERRIT_URL START_DATE END_DATE RESPONSE_WINDOW SLEEP_PER_REPO

printf '%s\n' "${REPOS[@]}" | xargs -I {} -P "$PARALLEL" bash -c 'fetch_one "$@"' _ {}

echo ""
echo "完了: $(ls "$RAW_JSON_DIR"/*.json 2>/dev/null | wc -l) / ${#REPOS[@]} repo"
echo ""
echo "[次の手順] per-repo を結合して combined を作る:"
echo "  ls $RAW_JSON_DIR/openstack__*.json > $OUT_BASE/raw_json_list.txt"
echo "  （combined CSV は build_dataset の per-repo CSV を pandas concat、もしくは"
echo "    既存 data/combined_raw_main32.csv の生成手順に合わせて結合）"

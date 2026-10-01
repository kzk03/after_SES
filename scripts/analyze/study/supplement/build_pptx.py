# -*- coding: utf-8 -*-
"""補足資料の pptx 組版と README 索引の生成.

a_feature_catalog.py / b_archetype.py / c_performance.py / d_setup.py が
書き出した tab/_manifest_*.json を読み、16:9 のスライドに並べる。

出力: docs/SES補足資料_自動生成.pptx
      outputs/analysis/ses_supplement/README.md

実行: .venv/bin/python scripts/analyze/study/supplement/build_pptx.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import FIG, OUT, ROOT, TAB  # noqa: E402

PPTX = ROOT / "docs" / "SES補足資料_自動生成.pptx"
GROUPS = ["a", "b", "c", "d"]
# スライドの並び順。manifest の生成順ではなく、このカテゴリ順に並べ替える。
# 各要素: (表示する章記号, 章タイトル, 章の一言, manifest 上の id を並べたもの)
# 表示 ID は章記号 + 通し番号に振り直す (元 ID は README に併記)。
CATEGORIES = [
    ("A", "A. 特徴量カタログ",
     "予測にも原型分析にも使う 37 次元の定義・単位・正規化",
     ["S-A1", "S-A2", "S-A3", "S-A4", "S-A5"]),
    ("B", "B. 原型分析の深掘り（ペア単位）",
     "定式化と K の決め方、開発者×ディレクトリのペアで見た 4 原型"
     "（K の決定・頑健性の図は dev/pair 両方を 1 枚に載せている）",
     ["S-B1", "S-B2", "S-B2t", "S-B2n",
      "S-B3", "S-B4", "S-B7", "S-B8",
      "S-B11", "S-B11t", "S-B12", "S-B13", "S-B13t", "S-B14", "S-B14t"]),
    ("C", "C. 原型分析の深掘り（開発者単位）",
     "開発者だけを単位にした 4 原型と、ペア単位の原型との対応",
     ["S-B5", "S-B6", "S-B9", "S-B10", "S-B15", "S-B15t"]),
    ("D", "D. 予測の深掘り",
     "IRL の定式化、原型別・期間別の AUC、有意差検定、アブレーションとアンサンブル",
     ["S-C10", "S-C1", "S-C1t", "S-C2", "S-C2t", "S-C3",
      "S-C4", "S-C4t", "S-C5", "S-C5t", "S-C6", "S-C7",
      "S-C8", "S-C8t", "S-C9", "S-C9t"]),
    ("E", "付録. 実験設定・データセット",
     "データ規模・ラベル定義・評価設計・学習設定",
     ["S-D1", "S-D2", "S-D3", "S-D4", "S-D5"]),
]
# 既存デッキ (docs/#2 SES発表用.pptx) と同じキャンバスサイズにする。
# スライド間のコピペでレイアウトが崩れないようにするため。
SLIDE_W, SLIDE_H = Emu(17337088), Emu(9752013)   # 18.96 x 10.66 inch (16:9)
SCALE = SLIDE_W / Inches(13.333)                 # 文字サイズの拡大率
FONT = "Hiragino Sans"   # macOS で確実に描画される日本語フォント
MAX_ROWS_PER_SLIDE = 16  # 1枚に載せる最大データ行数 (超えたら続きスライドに分割)
MAX_TOTAL_ROWS = 40      # これを超える表はスライドに載せず CSV を案内する
INK = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x77, 0x77, 0x77)
ACCENT = RGBColor(0xE6, 0x4B, 0x35)


def fx(frac: float) -> Emu:
    """スライド幅に対する割合 → EMU。"""
    return Emu(int(SLIDE_W * frac))


def fy(frac: float) -> Emu:
    """スライド高に対する割合 → EMU。"""
    return Emu(int(SLIDE_H * frac))


def pt(size: float) -> float:
    """13.333inch 基準で決めた文字サイズを実キャンバスに合わせる。"""
    return size * SCALE


def load_manifest() -> list[dict]:
    items = []
    for g in GROUPS:
        p = TAB / f"_manifest_{g}.json"
        if not p.exists():
            print(f"[warn] {p.name} が無い（該当スクリプトを先に実行する）")
            continue
        items.extend(json.loads(p.read_text(encoding="utf-8")))
    return items


def _parse_id(item_id: str) -> tuple[str, int, str]:
    """'S-B11t' → ('B', 11, 't')。想定外の形式は番号 0 として扱う。"""
    m = re.fullmatch(r"S-([A-Z])(\d+)([a-z]*)", item_id)
    if not m:
        return ("?", 0, "")
    return (m.group(1), int(m.group(2)), m.group(3))


def order_items(items: list[dict]) -> list[dict]:
    """manifest を CATEGORIES の順に並べ替え、表示 ID を振り直す。

    図と、その元数値の表 (S-B2 と S-B2t) は同じ番号を共有させる。
    どのカテゴリにも入っていない項目は末尾にまとめる (取りこぼし防止)。
    """
    by_id = {it["id"]: it for it in items if it["kind"] != "section"}
    used: set[str] = set()
    ordered: list[dict] = []
    for letter, title, message, ids in CATEGORIES:
        ordered.append({"id": f"S-{letter}0", "kind": "section",
                        "title": title, "message": message, "source": ""})
        counter, last = 0, None
        for old_id in ids:
            it = by_id.get(old_id)
            if it is None:
                print(f"[warn] {old_id} が manifest に無い（該当スクリプトを先に実行する）")
                continue
            _, num, suffix = _parse_id(old_id)
            if not (suffix and last == num):
                counter += 1
            last = num
            new = dict(it)
            new["old_id"] = old_id
            new["id"] = f"S-{letter}{counter}{suffix}"
            ordered.append(new)
            used.add(old_id)
    leftover = [it for it in items
                if it["kind"] != "section" and it["id"] not in used]
    if leftover:
        print(f"[warn] CATEGORIES 未記載のため末尾に置いた: "
              f"{[it['id'] for it in leftover]}")
        ordered.append({"id": "S-Z0", "kind": "section", "title": "その他",
                        "message": "カテゴリ未分類（build_pptx.py の CATEGORIES に追記する）",
                        "source": ""})
        for i, it in enumerate(leftover, start=1):
            new = dict(it)
            new["old_id"] = it["id"]
            new["id"] = f"S-Z{i}"
            ordered.append(new)
    return ordered


def _style(run, size: float, bold: bool = False, color: RGBColor = INK) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    # 日本語も同じフォントにする (latin だけだと游ゴシックにならない)
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.makeelement(qn(tag), {"typeface": FONT})
        rPr.append(el)


def _textbox(slide, text: str, left, top, width, height, size: float,
             bold: bool = False, color: RGBColor = INK, align_center=False):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align_center:
            from pptx.enum.text import PP_ALIGN
            p.alignment = PP_ALIGN.CENTER
        _style(p.add_run(), size, bold, color)
        p.runs[0].text = line
    return box


def add_title_slide(prs) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _textbox(s, "SES 発表 補足資料", fx(0.075), fy(0.32), fx(0.85),
             fy(0.16), pt(40), bold=True)
    _textbox(s, "逆強化学習を用いたモジュール別レビュー動態予測\n"
                "A. 特徴量カタログ ／ B. 原型分析（ペア単位） ／ "
                "C. 原型分析（開発者単位） ／ D. 予測の深掘り",
             fx(0.075), fy(0.49), fx(0.85), fy(0.16), pt(18), color=MUTED)
    _textbox(s, "図表はすべて outputs/ 配下の実出力から自動生成"
                "（scripts/analyze/study/supplement/）",
             fx(0.075), fy(0.85), fx(0.85), fy(0.07), pt(12), color=MUTED)


def add_section_slide(prs, item: dict) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _textbox(s, item["title"], fx(0.075), fy(0.39), fx(0.85),
             fy(0.13), pt(34), bold=True, color=ACCENT)
    _textbox(s, item["message"], fx(0.075), fy(0.53), fx(0.85),
             fy(0.11), pt(17), color=MUTED)


def _head(prs, item: dict):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _textbox(s, f"{item['id']}　{item['title']}", fx(0.034), fy(0.029),
             fx(0.93), fy(0.08), pt(24), bold=True)
    _textbox(s, item["message"], fx(0.034), fy(0.117), fx(0.93),
             fy(0.08), pt(13), color=MUTED)
    if item.get("source"):
        _textbox(s, f"出所: {item['source']}", fx(0.034), fy(0.927),
                 fx(0.93), fy(0.05), pt(9), color=MUTED)
    return s


def add_figure_slide(prs, item: dict) -> None:
    s = _head(prs, item)
    png = ROOT / item["path"]
    iw, ih = Image.open(png).size
    box_l, box_t = fx(0.034), fy(0.207)
    box_w, box_h = fx(0.93), fy(0.707)
    scale = min(box_w / iw, box_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    s.shapes.add_picture(str(png), Emu(int(box_l + (box_w - w) / 2)),
                         Emu(int(box_t + (box_h - h) / 2)), width=Emu(w),
                         height=Emu(h))


def add_table_slide(prs, item: dict) -> int:
    """表スライドを追加する。行数が多い表は分割し、極端に長い表は CSV を案内する。"""
    df = pd.read_csv(ROOT / item["path"]).fillna("")
    if len(df) > MAX_TOTAL_ROWS:
        s = _head(prs, item)
        _textbox(s, f"この表は {len(df):,} 行あるためスライドには載せていない。\n"
                    f"数値は次のCSVを参照: {item['path']}",
                 fx(0.034), fy(0.30), fx(0.93), fy(0.2), pt(16))
        return 1
    pages = [df.iloc[i:i + MAX_ROWS_PER_SLIDE]
             for i in range(0, len(df), MAX_ROWS_PER_SLIDE)] or [df]
    for idx, page in enumerate(pages):
        it = dict(item)
        if len(pages) > 1:
            it["title"] = f"{item['title']}（{idx + 1}/{len(pages)}）"
        _add_one_table(prs, it, page)
    return len(pages)


def _add_one_table(prs, item: dict, df: pd.DataFrame) -> None:
    s = _head(prs, item)
    n_rows, n_cols = len(df) + 1, len(df.columns)
    left, top = fx(0.034), fy(0.207)
    width, height = fx(0.93), Emu(int(fy(0.043) * n_rows))
    shape = s.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = shape.table
    widths = item.get("col_widths") or [1.0] * n_cols
    total = sum(widths)
    for j, w in enumerate(widths[:n_cols]):
        table.columns[j].width = Emu(int(width * w / total))
    size = pt(item.get("font_size") or 11)
    for j, c in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(c)
        _style(cell.text_frame.paragraphs[0].runs[0], size, bold=True)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, v in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(v)
            para = cell.text_frame.paragraphs[0]
            if not para.runs:
                para.add_run().text = ""
            _style(para.runs[0], size)
            cell.margin_top = cell.margin_bottom = Emu(9000)


def write_readme(items: list[dict], n_slides: int) -> None:
    lines = [
        "# SES 発表 補足資料",
        "",
        "`scripts/analyze/study/supplement/` の各スクリプトで自動生成した図表の索引。",
        "図表の数値はすべて実出力ファイルから読んでいる（出所列を参照）。",
        "",
        "## 再生成",
        "",
        "```",
        ".venv/bin/python scripts/analyze/study/supplement/a_feature_catalog.py",
        ".venv/bin/python scripts/analyze/study/supplement/b_archetype.py",
        ".venv/bin/python scripts/analyze/study/supplement/c_period_auc.py",
        ".venv/bin/python scripts/analyze/study/supplement/c_performance.py",
        ".venv/bin/python scripts/analyze/study/supplement/d_setup.py",
        ".venv/bin/python scripts/analyze/study/supplement/build_pptx.py",
        "```",
        "",
        "## 索引",
        "",
        "スライドは A. 特徴量カタログ → B. 原型分析（ペア単位） → "
        "C. 原型分析（開発者単位） → D. 予測の深掘り → 付録 の順に並べている"
        "（順番は `build_pptx.py` の `CATEGORIES` で定義）。",
        "旧ID は図表ファイル名（`fig/`・`tab/` の basename）。",
        "",
        "| ID | 旧ID | 種別 | タイトル | ファイル | 出所 |",
        "|---|---|---|---|---|---|",
    ]
    kind_ja = {"figure": "図", "table": "表", "section": "章扉"}
    for it in items:
        path = it.get("path", "")
        lines.append(
            f"| {it['id']} | {it.get('old_id', '—')} | {kind_ja[it['kind']]} | "
            f"{it['title']} | "
            f"{'`' + path + '`' if path else '—'} | {it.get('source', '') or '—'} |")
    lines += ["", f"スライド: `{PPTX.relative_to(ROOT)}`（{n_slides} 枚）", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[readme] {(OUT / 'README.md').relative_to(ROOT)}")


def main() -> None:
    items = order_items(load_manifest())
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    add_title_slide(prs)
    n = {"figure": 0, "table": 0, "section": 0}
    for it in items:
        if it["kind"] == "section":
            add_section_slide(prs, it)
        elif it["kind"] == "figure":
            add_figure_slide(prs, it)
        else:
            add_table_slide(prs, it)
        n[it["kind"]] += 1
    PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(PPTX)
    n_slides = len(prs.slides._sldIdLst)
    print(f"[pptx] {PPTX.relative_to(ROOT)}: {n_slides} 枚 "
          f"(図 {n['figure']} / 表 {n['table']} / 章扉 {n['section']} + 表紙1)")
    write_readme(items, n_slides)


if __name__ == "__main__":
    main()

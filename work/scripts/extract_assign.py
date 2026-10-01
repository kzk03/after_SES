"""割り当てイベントと「貢献したか」を抽出する。

イベント = 他人がレビュアーを追加した瞬間 (reviewer_updates, added_by != reviewer)。
貢献    = 追加時刻以降にメッセージまたは Code-Review 投票があったか (符号は問わない)。
          反応しない・投票せず外された (辞退) は非貢献。
"""
import json, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, "scripts/pipeline")
from build_dataset import FeatureBuilder

fb = FeatureBuilder()
def ts(s):
    return pd.Timestamp(s.split(".")[0]) if s else None

rows = []
for f in sorted(glob.glob("work/assign_json_full/*.json")):
    for ch in json.load(open(f)):
        proj = ch.get("project", ""); owner = ch.get("owner", {}).get("email", "")
        created = ts(ch.get("created"))
        if not created:
            continue
        acts: dict = {}
        for m in ch.get("messages", []):
            ae = m.get("author", {}).get("email", ""); md = ts(m.get("date"))
            if md and ae and ae != owner and not fb._is_bot(ae):
                acts.setdefault(ae, []).append(md)
        votes = {}
        for l in ((ch.get("labels") or {}).get("Code-Review", {}).get("all", []) or []):
            em = l.get("email"); v = l.get("value"); vd = ts(l.get("date"))
            if em and v not in (None, 0):
                votes[em] = v
                if vd:
                    acts.setdefault(em, []).append(vd)
        added, removed = {}, {}
        for ru in ch.get("reviewer_updates", []):
            rv = ru.get("reviewer", {}); em = rv.get("email", "")
            if not em or fb._is_bot(em) or "SERVICE_USER" in (rv.get("tags") or []):
                continue
            t = ts(ru.get("updated")); by = ru.get("updated_by", {}).get("email", "")
            if ru.get("state") == "REVIEWER":
                if em not in added or (t and t < added[em][0]):
                    added[em] = (t, by)
            elif ru.get("state") == "REMOVED":
                if em not in removed or (t and t < removed[em][0]):
                    removed[em] = (t, by)
        for em, (t_add, by) in added.items():
            if em == owner or by == em or not t_add:
                continue
            after = sorted(x for x in acts.get(em, []) if x >= t_add)
            first = after[0] if after else None
            rem = removed.get(em)
            rows.append({
                "project": proj, "change": int(ch.get("_number", 0)), "reviewer": em,
                "owner": owner, "added_at": t_add, "added_by": by,
                "by_owner": by == owner, "created": created, "first_act": first,
                "lat_days": (first - t_add).total_seconds() / 86400 if first is not None else np.nan,
                "voted": em in votes, "vote": votes.get(em),
                "removed_no_vote": (rem is not None) and (em not in votes),
            })

d = pd.DataFrame(rows).sort_values("added_at").reset_index(drop=True)
d.to_parquet("work/assign_events.parquet")
print(f"割り当てイベント {len(d):,} / {d.project.nunique()} プロジェクト / change {d.change.nunique():,}")
print(f"期間 {d.added_at.min():%Y-%m-%d} 〜 {d.added_at.max():%Y-%m-%d}")
print(f"オーナーが追加 {d.by_owner.mean():.4f}")
for h in (7, 14, 30):
    print(f"  貢献率 {h:>2}日: {(d.lat_days <= h).mean():.4f}")
print(f"  投票あり {d.voted.mean():.4f} (負 {(d.vote.fillna(0)<0).mean():.4f}) / 辞退 {d.removed_no_vote.mean():.4f}")

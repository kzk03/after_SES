"""AOSP の raw JSON から change → 触ったディレクトリ集合を作る。

path 特徴量 (柱1 の IRL 側で開発済み 6 次元) を柱2 の割り当て設計でも
使えるようにするための前処理。深さ 2 でディレクトリに集約する
(path_features.file_to_dir と同じ規約)。
"""
import glob, json, pickle, sys
sys.path.insert(0, "src")
from review_predictor.IRL.features.path_features import file_to_dir

m = {}
for f in sorted(glob.glob("work/assign_json_full/*.json")):
    d = json.load(open(f))
    n = 0
    for c in d:
        revs = c.get("revisions") or {}
        cur = c.get("current_revision")
        rv = revs.get(cur) or (list(revs.values())[0] if revs else None)
        if not rv:
            continue
        files = [p for p in (rv.get("files") or {}) if p not in ("/COMMIT_MSG", "/MERGE_LIST")]
        if not files:
            continue
        m[int(c["_number"])] = frozenset(file_to_dir(p) for p in files)
        n += 1
    print(f"  {f.split('/')[-1][:-5]:<40} {n:>7,} change")
pickle.dump(m, open("work/os_change_dirs.pkl", "wb"))
sz = sum(len(v) for v in m.values()) / max(len(m), 1)
print(f"\n保存 work/os_change_dirs.pkl: {len(m):,} change  平均 {sz:.1f} dir/change")

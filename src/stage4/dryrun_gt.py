"""HF GT 乾跑：16 場景的 GT layout 當 prototype → Stage 4 的 D.1/D.3/D.5，
在真實多樓層建築上驗證樓層合併、房間分割、門/開口、樓梯指派——不需 MP3D。
全域對照：論文統計 16 棟 / 33 層 / 317 房 / 34 樓梯。
"""
import glob
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/stage4")
sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/eval")
from gt_as_prototype import DATA, gt_prototype  # noqa: E402
from scene_graph import identify_levels, segment_rooms  # noqa: E402
from stairs import detect_stairs  # noqa: E402
from gt_loader import ALL_SCENES  # noqa: E402


def run_scene(scene: str):
    t0 = time.time()
    proto = gt_prototype(scene)
    levels = identify_levels(proto)
    n_rooms, n_doors, n_open = 0, 0, 0
    for lv in levels:
        segment_rooms(lv)
        n_rooms += len(lv.rooms)
        n_doors += sum(o["is_door"] for o in lv.openings)
        n_open += len(lv.openings)

    stair_files = sorted(glob.glob(f"{DATA}/stairs/{scene}/*.ply"))
    kept, rejected = 0, 0
    if stair_files:
        # 整場景樓梯檔合併成一個 mesh（flight+landing 由鄰近合併聚成「座」，同 D.5 語意）
        vs, ts, off = [], [], 0
        for f in stair_files:
            sm = o3d.io.read_triangle_mesh(f)
            v, t = np.asarray(sm.vertices), np.asarray(sm.triangles)
            if len(v) == 0:
                continue
            vs.append(v); ts.append(t + off); off += len(v)
        merged = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.concatenate(vs)),
            o3d.utility.Vector3iVector(np.concatenate(ts)))
        cands = detect_stairs(merged, levels)
        for c in cands:
            if c.rooms is not None:
                kept += 1
            else:
                rejected += 1
    from collections import Counter
    cls = Counter(p.cls for p in proto.polygons)
    print(f"{scene:>12s}  levels={len(levels)}  rooms={n_rooms:3d}  doors={n_doors:3d} "
          f"openings={n_open:3d}  stairs kept/rej={kept}/{rejected} (GT檔 {len(stair_files)})  "
          f"[fl={cls.get('floor',0)} ce={cls.get('ceiling',0)} wa={cls.get('wall',0)}] "
          f"({time.time()-t0:.0f}s)", flush=True)
    return len(levels), n_rooms, kept, len(stair_files)


if __name__ == "__main__":
    scenes = sys.argv[1:] if len(sys.argv) > 1 else ALL_SCENES
    tot = np.zeros(4, dtype=int)
    for s in scenes:
        try:
            tot += np.array(run_scene(s))
        except Exception as e:
            print(f"{s:>12s}  ERROR: {type(e).__name__}: {e}", flush=True)
    print(f"\n合計: levels={tot[0]} (論文 33)  rooms={tot[1]} (論文 317)  "
          f"stairs kept={tot[2]} / GT 檔 {tot[3]} (論文 34 座)")

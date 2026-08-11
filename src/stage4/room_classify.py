"""D.4 房型分類與室外剪枝（geometry env）：
openseg 點特徵 → 依 BEV 房間遮罩聚合成每房平均特徵 → cosine vs CLIP 文字嵌入 → 房型；
葉節點屬最後五類（室外）者剪除。更新 scene_graph.json。"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/home/ado/storage/HouseLayout3D/src/stage4")
from clip_text_emb import N_OUTDOOR_TAIL, ROOM_TYPES  # noqa: E402
from load_prototype import load_prototype  # noqa: E402
from scene_graph import RES, identify_levels, segment_rooms  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitted", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--openseg-dir", required=True)
    ap.add_argument("--text-emb", required=True)
    ap.add_argument("--scene-graph", required=True)
    args = ap.parse_args()

    pts = np.load(f"{args.openseg_dir}/openseg_points.npy")
    feats = np.load(f"{args.openseg_dir}/openseg_feats.npy").astype(np.float32)
    text = np.load(args.text_emb)

    proto = load_prototype(args.fitted, args.skeleton, args.probs, args.labels)
    levels = identify_levels(proto)
    with open(args.scene_graph) as f:
        graph = json.load(f)

    for lv, glv in zip(levels, graph["levels"]):
        segment_rooms(lv)
        origin = lv._grid_origin
        final = lv._final
        H, W = final.shape
        # 高度屬於本層的點
        zmask = (pts[:, 2] >= lv.elevation - 0.3) & (pts[:, 2] <= lv.elevation + 3.5)
        ij = ((pts[zmask, :2] - origin) / RES).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
        rid_of_pt = np.zeros(len(ij), dtype=int)
        rid_of_pt[ok] = final[ij[ok, 1], ij[ok, 0]]
        f_lv = feats[zmask]

        for room in glv["rooms"]:
            m = rid_of_pt == room["id"]
            if m.sum() < 50:
                room["type"] = "unknown"
                continue
            avg = f_lv[m].mean(axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-9)
            sims = text @ avg
            k = int(np.argmax(sims))
            room["type"] = ROOM_TYPES[k]
            room["type_scores"] = {ROOM_TYPES[i]: round(float(sims[i]), 4)
                                   for i in np.argsort(sims)[::-1][:5]}
            room["is_outdoor_class"] = bool(k >= len(ROOM_TYPES) - N_OUTDOOR_TAIL)

        # D.4 剪枝：葉節點（連通度 ≤1）且室外類 → 剪
        deg = {r["id"]: 0 for r in glv["rooms"]}
        for op in glv["openings"]:
            a, b = op["rooms"]
            if a in deg: deg[a] += 1
            if b in deg: deg[b] += 1
        pruned = [r["id"] for r in glv["rooms"]
                  if r.get("is_outdoor_class") and deg.get(r["id"], 0) <= 1]
        glv["pruned_rooms"] = pruned
        glv["rooms"] = [r for r in glv["rooms"] if r["id"] not in pruned]

    with open(args.scene_graph, "w") as f:
        json.dump(graph, f, indent=1)
    for glv in graph["levels"]:
        for r in glv["rooms"]:
            print(f"L{glv['idx']} room {r['id']}: type={r.get('type')} "
                  f"top5={list(r.get('type_scores', {}).items())[:3]}")
        if glv.get("pruned_rooms"):
            print(f"L{glv['idx']} 剪除室外葉節點: {glv['pruned_rooms']}")


if __name__ == "__main__":
    main()

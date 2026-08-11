"""Validate frozen split balance, derivation rule, and held-out guard."""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from house_floor_gt import build_scene  # noqa: E402
from split_io import load_split, select_scenes  # noqa: E402


ROOT = "/home/ado/storage/HouseLayout3D"
PATH = f"{ROOT}/configs/eval2d/split_v0_1.json"


def stats(scenes):
    levels = rooms = 0
    for scene in scenes:
        gt = build_scene(scene)
        levels += len(gt["lvl_z"])
        rooms += sum(room["in_roomset"] and not room.get("is_stairs")
                     for room in gt["rooms"])
    return {"scenes": len(scenes), "levels": levels, "rooms": rooms}


def main():
    split = load_split(PATH)
    checks = {
        "dev balance exact": stats(split["dev"]) == split["balance"]["dev"],
        "held-out balance exact": stats(split["held_out"]) == split["balance"]["held_out"],
        "total balance exact": stats(split["dev"] + split["held_out"])
                               == split["balance"]["total"],
        "dev selection allowed": select_scenes(split) == split["dev"],
    }
    try:
        select_scenes(split, "held_out")
        checks["held-out guard"] = False
    except PermissionError:
        checks["held-out guard"] = True
    checks["checkpoint allows held-out"] = select_scenes(
        split, "held_out", "fixture") == split["held_out"]

    derived_pairs = []
    rows = []
    for scene in split["dev"] + split["held_out"]:
        gt = build_scene(scene)
        rows.append((len(gt["lvl_z"]), sum(
            room["in_roomset"] and not room.get("is_stairs")
            for room in gt["rooms"]), scene))
    rows.sort()
    for index in range(0, len(rows), 2):
        a, b = rows[index:index + 2]
        if hashlib.sha256(a[2].encode()).hexdigest() < hashlib.sha256(b[2].encode()).hexdigest():
            dev, held = a, b
        else:
            dev, held = b, a
        derived_pairs.append((dev[2], held[2]))
    recorded_pairs = [(pair["dev"], pair["held_out"])
                      for pair in split["pairs"]]
    checks["selection rule reproducible"] = derived_pairs == recorded_pairs
    checks["selection excludes scores"] = not any(
        "score" in item.lower() for item in split["selection_inputs"])

    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"\n{sum(checks.values())}/{len(checks)} PASS")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen eval2d split loader with held-out access guard."""
from __future__ import annotations

import json


PARTITIONS = {"dev", "held_out", "all"}


def load_split(path):
    with open(path) as stream:
        split = json.load(stream)
    dev = list(split["dev"])
    held_out = list(split["held_out"])
    if len(dev) != 8 or len(held_out) != 8:
        raise ValueError("split must contain 8 dev and 8 held-out scenes")
    if set(dev) & set(held_out) or len(set(dev) | set(held_out)) != 16:
        raise ValueError("split must be a disjoint 16-scene partition")
    return split


def select_scenes(split, partition="dev", checkpoint_name=None):
    """Return scenes, rejecting accidental held-out access.

    A non-empty checkpoint name is an explicit audit trail, not a security
    mechanism.  Callers should persist it in their output manifest.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition}")
    if partition in {"held_out", "all"} and not checkpoint_name:
        raise PermissionError(
            "held-out access requires an explicit --checkpoint-name")
    if partition == "dev":
        return list(split["dev"])
    if partition == "held_out":
        return list(split["held_out"])
    return sorted(set(split["dev"]) | set(split["held_out"]))

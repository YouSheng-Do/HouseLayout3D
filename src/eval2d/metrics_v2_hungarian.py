"""Compatibility adapter for the archived ``eval2d_v2_hungarian`` protocol.

V2 used deterministic Hungarian matching but accidentally ignored rooms, doors,
and edges on unmatched predicted levels.  The implementation now lives in
``metrics.score_scene`` behind an explicit false flag so old frozen scores remain
reproducible without duplicating the evaluator.
"""
from metrics import *  # noqa: F401,F403
from metrics import score_scene as _score_scene

EVAL2D_VERSION = "eval2d_v2_hungarian"


def score_scene(gt, pred):
    return _score_scene(gt, pred, count_unmatched_pred_levels=False)

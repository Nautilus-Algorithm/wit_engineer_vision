"""pillar <-> exchange 实例配对: 把同一帧的多个部件实例拼成组合 schema 的完整点集。

tc 权重对每个实例都输出全部槽位, 但只有本类别负责的那段槽位有效, 其余为 padding。
组合 schema 的 ``parts`` 指明每个 class_id 负责哪些槽位; 本模块枚举
"每个部件取一个实例或缺省" 的组合, 对每个组合解 PnP, 选内点最多、误差最小的一组。
错配的组合 (柱子和兑换口来自不同兑换站) 几何上不一致, 会因内点少或重投影误差大落选。
"""

from dataclasses import dataclass
from itertools import product
from typing import Optional, Sequence

import numpy as np

from .schema import KeypointSchema
from .solver import PnPEstimator, PoseEstimate


@dataclass(frozen=True)
class PartObservation:
    """一个 detector 实例; points/confidence 覆盖 schema 的全部槽位。"""

    class_id: int
    points: np.ndarray       # (N, 2)
    confidence: np.ndarray   # (N,)


@dataclass(frozen=True)
class Candidate:
    points: np.ndarray       # (N, 2)
    confidence: np.ndarray   # (N,), 未取自任何实例的槽位为 0
    sources: tuple[Optional[int], ...]  # 按 sorted(schema.parts) 排列的实例下标, None 表示缺省
    score: float


@dataclass(frozen=True)
class Solution:
    candidate: Optional[Candidate]
    estimate: Optional[PoseEstimate]
    reasons: tuple[str, ...]


def merge_candidates(schema: KeypointSchema, observations: Sequence[PartObservation]) -> list[Candidate]:
    count = len(schema.names)
    class_ids = sorted(schema.parts)
    by_class: dict[int, list[int]] = {cid: [] for cid in class_ids}
    for index, obs in enumerate(observations):
        points = np.asarray(obs.points, dtype=np.float64)
        confidence = np.asarray(obs.confidence, dtype=np.float64).reshape(-1)
        if points.shape != (count, 2) or confidence.shape != (count,):
            raise ValueError(f"observation {index} must cover all {count} schema slots")
        if obs.class_id in by_class:
            by_class[obs.class_id].append(index)

    candidates = []
    for sources in product(*([None, *by_class[cid]] for cid in class_ids)):
        if all(source is None for source in sources):
            continue
        points = np.zeros((count, 2), dtype=np.float64)
        confidence = np.zeros(count, dtype=np.float64)
        for cid, source in zip(class_ids, sources):
            if source is None:
                continue
            slots = list(schema.parts[cid])
            points[slots] = np.asarray(observations[source].points, dtype=np.float64)[slots]
            confidence[slots] = np.asarray(observations[source].confidence, dtype=np.float64)[slots]
        candidates.append(Candidate(points, confidence, tuple(sources), float(confidence.sum())))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def solve_best_candidate(schema: KeypointSchema, observations: Sequence[PartObservation],
                         estimator: PnPEstimator, camera_matrix, distortion_coeffs) -> Solution:
    candidates = merge_candidates(schema, observations)
    if not candidates:
        return Solution(None, None, ("no observation matches the schema parts",))
    best: Optional[tuple[Candidate, PoseEstimate]] = None
    reasons = []
    for candidate in candidates:
        estimate = estimator.estimate(schema.object_points, candidate.points,
                                      camera_matrix, distortion_coeffs, candidate.confidence)
        if not estimate.valid:
            reasons.append(f"{candidate.sources}: {estimate.reason}")
            continue
        key = (int(estimate.inlier_mask.sum()), -estimate.reprojection_error_px)
        if best is None or key > (int(best[1].inlier_mask.sum()), -best[1].reprojection_error_px):
            best = (candidate, estimate)
    if best is None:
        return Solution(None, None, tuple(reasons))
    return Solution(best[0], best[1], tuple(reasons))

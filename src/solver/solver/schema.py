"""数据驱动的目标关键点 schema。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class KeypointSchema:
    name: str
    class_id: int
    names: tuple[str, ...]
    object_points: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.object_points, dtype=np.float64)
        if points.ndim != 2 or points.shape != (len(self.names), 3):
            raise ValueError("object_points must have shape (len(names), 3)")
        if not np.isfinite(points).all():
            raise ValueError("object points must be finite")
        points = np.ascontiguousarray(points)
        points.setflags(write=False)
        object.__setattr__(self, "object_points", points)

    def points_for_observation(self, count: int) -> np.ndarray:
        if count != len(self.names):
            raise ValueError(f"observation count {count} does not match schema count {len(self.names)}")
        return self.object_points


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_schema(path: str | Path, name: str) -> KeypointSchema:
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    root = _as_mapping(document, "schema")
    if name not in root:
        raise KeyError(f"schema {name!r} missing")
    entry = _as_mapping(root[name], f"schema {name}")
    keypoints = entry.get("keypoints")
    if not isinstance(keypoints, dict) or not keypoints:
        raise ValueError(f"schema {name} keypoints must be a non-empty mapping")
    try:
        class_id = int(entry["class_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"schema {name} class_id must be an integer") from exc

    names = tuple(str(point_name) for point_name in keypoints)
    rows = []
    for point_name, value in keypoints.items():
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"keypoint {point_name} must be a length-3 sequence")
        try:
            point = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"keypoint {point_name} must be numeric") from exc
        if not np.isfinite(point).all():
            raise ValueError(f"keypoint {point_name} must be finite")
        rows.append(point)
    return KeypointSchema(name, class_id, names, np.asarray(rows, dtype=np.float64))

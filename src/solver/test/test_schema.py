from pathlib import Path

import numpy as np
import pytest

from solver.schema import KeypointSchema, load_schema


def test_load_schema_rejects_empty_todo_entry(tmp_path: Path):
    path = tmp_path / "schema.yaml"
    path.write_text("exchange: {class_id: 1, keypoints: {}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="keypoints"):
        load_schema(path, "exchange")


def test_load_schema_preserves_order_and_returns_float64_points(tmp_path: Path):
    path = tmp_path / "schema.yaml"
    path.write_text(
        """exchange:
  class_id: 7
  keypoints:
    TL: [-0.1, -0.2, 0.3]
    ring: [0.1, 0.2, 0.4]
""",
        encoding="utf-8",
    )

    schema = load_schema(path, "exchange")

    assert isinstance(schema, KeypointSchema)
    assert schema.name == "exchange"
    assert schema.class_id == 7
    assert schema.names == ("TL", "ring")
    assert schema.object_points.dtype == np.float64
    assert schema.object_points.flags.c_contiguous
    np.testing.assert_allclose(schema.points_for_observation(2), schema.object_points)


def test_schema_rejects_unknown_name_and_bad_observation_count(tmp_path: Path):
    path = tmp_path / "schema.yaml"
    path.write_text(
        """exchange:
  class_id: 7
  keypoints:
    TL: [0, 0, 1]
    TR: [1, 0, 1]
    BL: [0, 1, 1]
    BR: [1, 1, 1]
""",
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="missing"):
        load_schema(path, "missing")
    schema = load_schema(path, "exchange")
    with pytest.raises(ValueError, match="count"):
        schema.points_for_observation(3)


def test_schema_rejects_nonfinite_point(tmp_path: Path):
    path = tmp_path / "schema.yaml"
    path.write_text(
        """exchange:
  class_id: 7
  keypoints:
    TL: [nan, 0, 1]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite"):
        load_schema(path, "exchange")

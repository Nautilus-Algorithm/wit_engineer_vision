import numpy as np
import pytest

from solver.pnp_node import _FrameBuffer, _camera_info_to_arrays, _observation_to_arrays


class Header:
    frame_id = "camera_optical"


class CameraInfo:
    k = list(range(9))
    d = [0.1, 0.2, 0.3, 0.4, 0.5]
    header = Header()


class Observation:
    schema = "exchange"
    class_id = 3
    u = [10.0, 20.0]
    v = [11.0, 21.0]
    confidence = [0.9, 0.8]


def test_camera_info_conversion():
    K, D = _camera_info_to_arrays(CameraInfo())
    assert K.shape == (3, 3)
    assert D.shape == (5,)
    np.testing.assert_allclose(K[0], [0, 1, 2])


def test_observation_conversion_preserves_parallel_arrays():
    schema, class_id, points, confidence = _observation_to_arrays(Observation())
    assert schema == "exchange"
    assert class_id == 3
    assert points.shape == (2, 2)
    assert confidence.shape == (2,)
    np.testing.assert_allclose(points[:, 0], [10.0, 20.0])


@pytest.mark.parametrize("field", ["v", "confidence"])
def test_observation_conversion_rejects_mismatched_arrays(field):
    message = Observation()
    setattr(message, field, [1.0])
    with pytest.raises(ValueError, match="parallel"):
        _observation_to_arrays(message)


def test_frame_buffer_releases_frame_when_newer_stamp_arrives():
    buffer = _FrameBuffer(timeout_s=0.05)
    assert buffer.add((1, 0), "a", now=0.00) == []
    assert buffer.add((1, 0), "b", now=0.01) == []
    assert buffer.add((2, 0), "c", now=0.02) == [((1, 0), ["a", "b"])]
    assert buffer.flush(now=0.03) == []
    assert buffer.flush(now=0.08) == [((2, 0), ["c"])]


def test_frame_buffer_drops_late_message_of_released_frame():
    buffer = _FrameBuffer(timeout_s=0.05)
    buffer.add((2, 0), "new", now=0.0)
    assert buffer.add((1, 0), "late", now=0.01) == []
    assert buffer.flush(now=1.0) == [((2, 0), ["new"])]


def test_camera_info_conversion_rejects_bad_intrinsics():
    message = CameraInfo()
    message.k = [1.0] * 8
    with pytest.raises(ValueError, match="K"):
        _camera_info_to_arrays(message)

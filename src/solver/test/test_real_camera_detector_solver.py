import os
import shutil
import subprocess

import pytest


@pytest.mark.skipif(os.environ.get("RUN_REAL_CAMERA_TEST") != "1",
                    reason="set RUN_REAL_CAMERA_TEST=1 to require real camera and weights")
def test_real_camera_detector_solver_dataflow():
    """Opt-in smoke test; enabled runs must expose missing hardware/config as failures."""
    for executable in ("ros2",):
        if shutil.which(executable) is None:
            pytest.fail(f"required executable not found: {executable}")
    model = os.environ.get("REAL_DETECTOR_MODEL", "")
    if model and not os.path.exists(model):
        pytest.fail(f"REAL_DETECTOR_MODEL does not exist: {model}")
    timeout = float(os.environ.get("REAL_TIMEOUT_S", "10"))
    command = ["ros2", "run", "camera", "camera_node"]
    camera = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        try:
            output, _ = camera.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            camera.terminate()
            output, _ = camera.communicate(timeout=3)
        if camera.returncode not in (0, -15):
            pytest.fail(f"camera_node failed with code {camera.returncode}:\n{output}")
        pytest.fail("real camera smoke test did not receive detector/solver observations")
    finally:
        if camera.poll() is None:
            camera.kill()
            camera.wait()

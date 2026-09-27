"""rm26_arm 专用闭式 IK (``arm.ik_type: rm26_arm``) 的回归测试。

推导、分支编号与实测见 docs/2026-09-26-rm26-arm-wrist-structure-and-closed-form-ik.md
第四、九节。 闭式解本身假设 d[3]=0, 精度测试在 d[3]=0 的模型上做; 真模型
(d[3]=0.5 mm) 只测"闭式初值 + 修正/精修"的收敛率。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from planning import ArmModel, load_config


@pytest.fixture(scope="module")
def real_arm() -> ArmModel:
    return ArmModel.from_config(dict(load_config()["arm"], ik_type="rm26_arm"))


@pytest.fixture(scope="module")
def gap_free_arm(real_arm: ArmModel) -> ArmModel:
    d = real_arm.d.copy()
    d[3] = 0.0
    return dataclasses.replace(real_arm, d=d)


def _sample(arm: ArmModel, count: int, seed: int) -> np.ndarray:
    lower, upper = arm.joint_space.lower, arm.joint_space.upper
    return lower + np.random.default_rng(seed).random((count, 6)) * (upper - lower)


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _tcp_errors(arm: ArmModel, joints: np.ndarray, targets: np.ndarray):
    """joints (B, K, 6) 对 targets (B, 4, 4) 的位置 / 旋转误差, 形状 (B, K)。"""
    count, width = joints.shape[:2]
    tcp = arm.forward_kinematics(joints.reshape(-1, 6)).tcp_transforms.reshape(count, width, 4, 4)
    position = np.linalg.norm(tcp[..., :3, 3] - targets[:, None, :3, 3], axis=-1)
    relative = np.swapaxes(targets[:, None, :3, :3], -1, -2) @ tcp[..., :3, :3]
    rotation = np.linalg.norm(
        Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec(), axis=-1
    ).reshape(count, width)
    return position, rotation


def _assert_solved(arm: ArmModel, joints_sampled: np.ndarray, tolerance: float):
    targets = arm.forward_kinematics(joints_sampled).tcp_transforms
    joints, valid = arm.solve_ik(targets)
    assert joints.shape == (len(targets), 8, 6)
    position, rotation = _tcp_errors(arm, joints, targets)
    # valid 的解必须真的复现目标
    assert np.all(position[valid] < tolerance)
    assert np.all(rotation[valid] < tolerance)
    joint_space = arm.joint_space
    inside = (joints >= joint_space.lower - 1e-12) & (joints <= joint_space.upper + 1e-12)
    assert np.all(inside.all(axis=-1)[valid])
    return joints, valid


def test_golden_round_trip_gap_free(gap_free_arm: ArmModel):
    """FK -> IK -> FK: d[3]=0 模型下每例都解回采样那组关节, 精度 1e-9。"""
    sampled = _sample(gap_free_arm, 2000, seed=0)
    joints, valid = _assert_solved(gap_free_arm, sampled, 1e-9)
    assert valid.any(axis=1).all()
    matches = valid & (np.max(np.abs(_wrap(joints - sampled[:, None])), axis=-1) < 1e-6)
    assert matches.any(axis=1).all(), "有目标没解回采样的那组关节"
    # 分支分布 (文档 9.3): 真解主要在 0 和 4, 各约一半
    histogram = np.bincount(np.argmax(matches, axis=1), minlength=8)
    assert histogram[0] > 0.4 * len(sampled)
    assert histogram[4] > 0.4 * len(sampled)


def test_branch_numbering(gap_free_arm: ArmModel):
    """branch = 4·[s6<0] + 2·[s2<0] + [s4<0]: 同 s6 的 4 支共用 θ6, 同 (s6, s2) 共用 θ1, θ2。"""
    sampled = _sample(gap_free_arm, 200, seed=1)
    targets = gap_free_arm.forward_kinematics(sampled).tcp_transforms
    joints, _ = gap_free_arm.solve_ik(targets)
    for group in ((0, 1, 2, 3), (4, 5, 6, 7)):
        spread = _wrap(joints[:, group, 5] - joints[:, group[:1], 5])
        np.testing.assert_allclose(spread, 0.0, atol=1e-9)
    for pair in ((0, 1), (2, 3), (4, 5), (6, 7)):
        np.testing.assert_allclose(
            _wrap(joints[:, pair[0], :2] - joints[:, pair[1], :2]), 0.0, atol=1e-9
        )


def test_q5_above_pi(gap_free_arm: ArmModel):
    """q5 限位 [-1.39, 4.36] 跨过 π: q5 ∈ (π, 4.36] 的构型必须能解出 (文档 9.4)。"""
    sampled = _sample(gap_free_arm, 300, seed=2)
    sampled[:, 4] = np.random.default_rng(3).uniform(np.pi, gap_free_arm.joint_space.upper[4], 300)
    _, valid = _assert_solved(gap_free_arm, sampled, 1e-9)
    assert valid.any(axis=1).all()


def test_wrist_singularity(gap_free_arm: ArmModel):
    """θ4 = −π (q4 = π − theta_offset[3] ≈ 0.00607) 时 θ3/θ5 只定和, 仍要有解且落限位。"""
    sampled = _sample(gap_free_arm, 300, seed=4)
    sampled[:, 3] = np.pi - gap_free_arm.theta_offset[3]
    _, valid = _assert_solved(gap_free_arm, sampled, 1e-9)
    assert valid.any(axis=1).all()


def test_real_model_convergence(real_arm: ArmModel):
    """真模型 d[3]=0.5 mm: 闭式初值 + 修正/精修, 有效解率 ≥ 99.9% (1000 例允许 1 例失败)。"""
    sampled = _sample(real_arm, 1000, seed=5)
    _, valid = _assert_solved(real_arm, sampled, 1e-6)
    assert valid.any(axis=1).sum() >= 999


def test_unreachable_target_is_invalid(real_arm: ArmModel):
    target = np.eye(4)
    target[:3, 3] = [2.0, 0.0, 0.0]
    joints, valid = real_arm.solve_ik(target[None])
    assert joints.shape == (1, 8, 6)
    assert not valid.any()


def test_branch_subset_matches_full(real_arm: ArmModel):
    sampled = _sample(real_arm, 50, seed=6)
    targets = real_arm.forward_kinematics(sampled).tcp_transforms
    full_joints, full_valid = real_arm.solve_ik(targets)
    joints, valid = real_arm.solve_ik(targets, branches=(4, 0))
    np.testing.assert_array_equal(valid, full_valid[:, [4, 0]])
    np.testing.assert_allclose(joints[valid], full_joints[:, [4, 0]][valid], atol=1e-9)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"ik_type": "rm26_arm"}, "rm26_arm"),
        ({"ik_type": "spherical_wrist"}, "spherical_wrist"),
        ({"ik_type": "numeric"}, "numeric"),
        ({"ik_type": None, "use_analytic_ik": True}, "spherical_wrist"),
        ({"ik_type": None, "use_analytic_ik": False}, "numeric"),
    ],
)
def test_ik_type_selection(overrides: dict, expected: str):
    config = dict(load_config()["arm"])
    config.pop("ik_type", None)
    config.update({key: value for key, value in overrides.items() if value is not None})
    assert ArmModel.from_config(config).ik_type == expected


def test_invalid_ik_type_rejected():
    with pytest.raises(ValueError, match="ik_type"):
        ArmModel.from_config(dict(load_config()["arm"], ik_type="nope"))


def test_config_selects_closed_form():
    assert ArmModel.from_config(load_config()["arm"]).ik_type == "rm26_arm"


def test_spherical_wrist_still_solves_spherical_arm():
    """保留的标准球腕解析 IK 仍可选, 且对真正的球腕臂 (末三轴共点) 能闭环。"""
    # config/planning.yaml 注释里的参考臂 (J4/J5/J6 共点), 其余字段沿用真 config
    arm = ArmModel.from_config(
        dict(
            load_config()["arm"],
            ik_type="spherical_wrist",
            dh={
                "alpha": [0.0, -np.pi / 2, np.pi, -np.pi / 2, np.pi / 2, -np.pi / 2],
                "a": [0.0, 0.0, 0.38, 0.0, 0.0, 0.0],
                "d": [0.0, 0.0, 0.0, 0.4345, 0.0, 0.0],
                "theta_offset": [0.0, -2.7525588, 1.7870426, -np.pi, 0.0, 0.0],
            },
            joint_limits={"lower": [-np.pi] * 6, "upper": [np.pi] * 6},
        )
    )
    sampled = np.random.default_rng(7).uniform(-2.5, 2.5, (200, 6))
    targets = arm.forward_kinematics(sampled).tcp_transforms
    joints, valid = arm.solve_ik(targets)
    position, rotation = _tcp_errors(arm, joints, targets)
    assert valid.any(axis=1).mean() > 0.95
    assert np.all(position[valid] < 1e-6) and np.all(rotation[valid] < 1e-6)

"""arm_model.py — 机械臂运动学/动力学模型 (整个规划库的核心)。

【这个文件干什么】
用一张 DH 表描述这台 6 轴臂的几何, 提供三个能力:
  1. forward_kinematics (正运动学, FK): 给 6 个关节角 -> 算出末端 TCP
     在 arm_base 系的 4x4 位姿。 "关节 -> 位姿", 唯一解, 简单。
  2. solve_ik (逆运动学, IK): 给想要的末端位姿 -> 反解出关节角。
     "位姿 -> 关节", 多解。 这里用解析法, 一次给出最多 8 组解 (称 8 个
     "分支", 对应肘上/肘下、腕翻转等不同姿态), 并标出哪些在限位内。
  3. inverse_dynamics (逆动力学, RNEA): 给运动状态算关节力矩 (力控/前馈
     用, 本项目的纯轨迹规划暂不强依赖)。

【要懂的概念 —— 也是你最该关注的一点】
- DH 表 (Denavit-Hartenberg) 是描述串联臂几何的一组参数:
  每个关节用 4 个数 (a, alpha, d, theta_offset) 描述它相对上一节的位置和
  朝向。 6 个关节就是 6 组这样的数, 外加末端工具偏置 tcp_offset。
  ⚠️ 本文件 FK/IK 用的是 **Modified (Craig) DH** 约定 (见 _forward_dh_transforms
  的矩阵形式), 不是经典/标准 DH —— 建表时别把两套约定搞混 (a、alpha 差一位下标)。
- **DH 表就是这台臂的"身份证"。 FK/IK 全靠它。 换一台臂 = 换 DH 表。**
  当前 config/planning.example.yaml 里的 DH 还是参考项目那台臂的, 必须
  按你的 rm26_arm.urdf 重新推导后覆盖, 否则 FK/IK 算出来的位姿是错的。
  (怎么从 URDF 得到 DH, 见 planning/README.md 的"如何确认与移植"。)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .joint_space import JointSpace
from .transform import validate_transforms


IK_TYPES = ("spherical_wrist", "rm26_arm", "numeric")


@dataclass(frozen=True, slots=True)
class ForwardKinematics:
    link_transforms: np.ndarray
    tcp_transforms: np.ndarray

    @property
    def link_origins(self) -> np.ndarray:
        return self.link_transforms[:, :, :3, 3]

    @property
    def link_rotations(self) -> np.ndarray:
        return self.link_transforms[:, :, :3, :3]


@dataclass(frozen=True, slots=True)
class ArmModel:
    a: np.ndarray
    alpha: np.ndarray
    d: np.ndarray
    theta_offset: np.ndarray
    tool_offset: np.ndarray
    joint_space: JointSpace
    masses: np.ndarray
    centers_of_mass: np.ndarray
    inertias: np.ndarray
    use_analytic_ik: bool = True
    # IK 解法, 见 IK_TYPES。 None = 按 use_analytic_ik 映射 (True->spherical_wrist, False->numeric)。
    ik_type: str | None = None

    def __post_init__(self) -> None:
        for name in ("a", "alpha", "d", "theta_offset"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (6,):
                raise ValueError(f"{name} must have shape (6,), got {values.shape}")
            object.__setattr__(self, name, values)
        tool_offset = np.asarray(self.tool_offset, dtype=float)
        masses = np.asarray(self.masses, dtype=float)
        centers = np.asarray(self.centers_of_mass, dtype=float)
        inertias = np.asarray(self.inertias, dtype=float)
        if tool_offset.shape != (3,):
            raise ValueError(f"tool_offset must have shape (3,), got {tool_offset.shape}")
        if masses.shape != (6,) or centers.shape != (6, 3) or inertias.shape != (6, 3, 3):
            raise ValueError("dynamics parameters must have shapes (6,), (6, 3), and (6, 3, 3)")
        if self.joint_space.dof != 6:
            raise ValueError("ArmModel requires a six-dimensional JointSpace")
        object.__setattr__(self, "tool_offset", tool_offset)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "centers_of_mass", centers)
        object.__setattr__(self, "inertias", inertias)
        object.__setattr__(self, "use_analytic_ik", bool(self.use_analytic_ik))
        ik_type = self.ik_type
        if ik_type is None:
            ik_type = "spherical_wrist" if self.use_analytic_ik else "numeric"
        if ik_type not in IK_TYPES:
            raise ValueError(f"ik_type must be one of {IK_TYPES}, got {ik_type!r}")
        object.__setattr__(self, "ik_type", ik_type)

    @classmethod
    def from_config(cls, config: dict) -> "ArmModel":
        dh = config["dh"]
        dynamics = config["dynamics"]
        if len(dynamics) != 6:
            raise ValueError("arm.dynamics must describe six links")
        return cls(
            a=np.asarray(dh["a"], dtype=float),
            alpha=np.asarray(dh["alpha"], dtype=float),
            d=np.asarray(dh["d"], dtype=float),
            theta_offset=np.asarray(dh["theta_offset"], dtype=float),
            tool_offset=np.asarray(config["tool"]["tcp_offset_6_tcp"], dtype=float),
            joint_space=JointSpace.from_config(config["joint_limits"]),
            masses=np.asarray([link["mass"] for link in dynamics], dtype=float),
            centers_of_mass=np.asarray([link["ipos"] for link in dynamics], dtype=float),
            inertias=np.asarray([link["inertia"] for link in dynamics], dtype=float),
            # config 的 arm.ik_type 优先; 没写时退回旧开关 arm.use_analytic_ik
            # (true=spherical_wrist, false=numeric, 缺省 true), 向后兼容。
            use_analytic_ik=bool(config.get("use_analytic_ik", True)),
            ik_type=config.get("ik_type"),
        )

    def solve_ik(
        self,
        target_tcp: np.ndarray,
        branches: tuple[int, ...] | list[int] | None = None,
        *,
        reference_joints: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """逆运动学统一入口: 给一批 ``(B, 4, 4)`` TCP 目标, 反解关节角。

        按 ``ik_type`` (来自 config 的 ``arm.ik_type``) 分派:
          - ``spherical_wrist`` -> :meth:`_solve_ik_analytic`, 末三轴共点的标准球腕臂;
          - ``rm26_arm``        -> :meth:`_solve_ik_closed_form`, J3/J4/J5 共点、腕心在 J4 的
            rm26_arm 专用闭式解 (推导见 docs/2026-09-26-rm26-arm-wrist-structure-and-closed-form-ik.md);
          - ``numeric``         -> :meth:`_solve_ik_numeric`, scipy least_squares 通用兜底。
        三者返回同样的契约: ``joints`` 形如 ``(B, n_branches, 6)``,
        ``valid`` 形如 ``(B, n_branches)`` —— 调用方 (type3 等) 无需关心用了哪种。
        分支号的含义随解法而不同 (见各方法 docstring), 换 ``ik_type`` 要同步 config 里的 ``ik_branches``。

        ``reference_joints`` (``(6,)`` 或 ``(B, 6)``, 可选) 目前只有 ``rm26_arm`` 使用:
        腕奇异时 θ3/θ5 只能定和, 用它的 q3 补齐; 不给就取 q3 限位中点。
        """
        if self.ik_type == "rm26_arm":
            return self._solve_ik_closed_form(target_tcp, branches, reference_joints=reference_joints)
        if self.ik_type == "spherical_wrist":
            return self._solve_ik_analytic(target_tcp, branches)
        return self._solve_ik_numeric(target_tcp, branches)

    def _solve_ik_analytic(
        self,
        target_tcp: np.ndarray,
        branches: tuple[int, ...] | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """解析 IK (8 分支, 假设球腕): 给 ``(B, 4, 4)`` TCP 目标反解关节角。

        只适用于末三轴 (J4/J5/J6) 共点的标准球腕臂 (``ik_type: spherical_wrist``)。
        ⚠️ 不适用于 rm26_arm: 它共点的是 J3/J4/J5, 用 ``ik_type: rm26_arm``。
        分支号 = ``2 * sign_index + wrist_flip``。
        """
        targets = validate_transforms(target_tcp, name="target_tcp")
        branch_ids = self._normalize_branches(branches)
        rotations = targets[:, :3, :3]
        positions = targets[:, :3, 3] - np.einsum("bij,j->bi", rotations, self.tool_offset)
        batch_size = len(targets)
        joints = np.zeros((batch_size, len(branch_ids), 6), dtype=float)
        valid = np.zeros((batch_size, len(branch_ids)), dtype=bool)

        a2 = self.a[2]
        a3 = self.a[3]
        d4 = self.d[3]
        coefficient_a = 2.0 * a2 * a3
        coefficient_b = -2.0 * a2 * d4
        x, y, z = positions.T
        radius_squared = np.sum(positions * positions, axis=1)
        workspace_valid = (radius_squared >= 0.055**2) & (radius_squared <= 0.8464)
        coefficient_c = a2**2 + a3**2 + d4**2 - radius_squared
        discriminant = coefficient_b**2 - coefficient_c**2 + coefficient_a**2
        workspace_valid &= discriminant >= -1e-9
        theta1 = np.arctan2(y, x) - np.pi
        theta1_flipped = theta1 + np.pi
        branch_output = {branch: index for index, branch in enumerate(branch_ids)}
        signs = ((1, 1), (-1, 1), (1, -1), (-1, -1))

        for sign_index in sorted({branch // 2 for branch in branch_ids}):
            sign1, sign2 = signs[sign_index]
            theta3 = 2.0 * np.arctan2(
                -(coefficient_b + sign1 * np.sqrt(np.maximum(discriminant, 0.0))),
                -(coefficient_a - coefficient_c),
            )
            f1 = a3 * np.cos(theta3) - d4 * np.sin(theta3) + a2
            f2 = -a3 * np.sin(theta3) - d4 * np.cos(theta3)
            second_discriminant = f1**2 + f2**2 - z**2
            theta2 = 2.0 * np.arctan2(
                f1 + sign2 * np.sqrt(np.maximum(second_discriminant, 0.0)),
                f2 - z,
            )
            radial = np.cos(theta2) * f1 - np.sin(theta2) * f2
            tolerance = 1e-4
            first_valid = (
                (np.abs(np.cos(theta1) * radial - x) < tolerance)
                & (np.abs(np.sin(theta1) * radial - y) < tolerance)
            )
            flipped_valid = (
                (np.abs(np.cos(theta1_flipped) * radial - x) < tolerance)
                & (np.abs(np.sin(theta1_flipped) * radial - y) < tolerance)
            )
            theta1_selected = np.where(first_valid, theta1, theta1_flipped)

            inverse_10 = self._inverse_dh_rotations(theta1_selected, self.alpha[0])
            inverse_21 = self._inverse_dh_rotations(theta2, self.alpha[1])
            inverse_32 = self._inverse_dh_rotations(theta3, self.alpha[2])
            inverse_30 = inverse_32 @ inverse_21 @ inverse_10
            rotation_36 = inverse_30 @ rotations
            singular = (np.abs(rotation_36[:, 0, 2]) < 1e-5) & (
                np.abs(rotation_36[:, 2, 2]) < 1e-5
            )
            theta4 = np.where(
                singular,
                0.0,
                np.arctan2(rotation_36[:, 2, 2], -rotation_36[:, 0, 2]),
            )
            rotation_46 = self._inverse_dh_rotations(theta4, self.alpha[3]) @ rotation_36
            base_solution = np.stack(
                (
                    theta1_selected - self.theta_offset[0],
                    theta2 - self.theta_offset[1],
                    theta3 - self.theta_offset[2],
                    theta4 - self.theta_offset[3],
                    np.arctan2(-rotation_46[:, 0, 2], rotation_46[:, 2, 2]) - self.theta_offset[4],
                    np.arctan2(rotation_46[:, 1, 0], rotation_46[:, 1, 1]) - self.theta_offset[5],
                ),
                axis=1,
            )
            wrist_flipped = base_solution.copy()
            wrist_flipped[:, 3] += np.pi
            wrist_flipped[:, 4] = -base_solution[:, 4] - 2.0 * self.theta_offset[4]
            wrist_flipped[:, 5] += np.pi
            base_solution, base_limits_valid = self.joint_space.normalize(base_solution)
            wrist_flipped, flipped_limits_valid = self.joint_space.normalize(wrist_flipped)
            common_valid = workspace_valid & (second_discriminant >= -1e-9) & (first_valid | flipped_valid)

            base_branch = 2 * sign_index
            if base_branch in branch_output:
                output_index = branch_output[base_branch]
                joints[:, output_index] = base_solution
                valid[:, output_index] = common_valid & base_limits_valid
            if base_branch + 1 in branch_output:
                output_index = branch_output[base_branch + 1]
                joints[:, output_index] = wrist_flipped
                valid[:, output_index] = common_valid & flipped_limits_valid
        return joints, valid

    def _solve_ik_closed_form(
        self,
        target_tcp: np.ndarray,
        branches: tuple[int, ...] | list[int] | None = None,
        *,
        reference_joints: np.ndarray | None = None,
        singular_tolerance: float = 1e-6,
        position_tolerance: float = 1e-6,
        rotation_tolerance: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """rm26_arm 专用闭式 IK (``ik_type: rm26_arm``), 目标在 ``arm_base`` 系 (line 已知)。

        rm26_arm 的 J3/J4/J5 三轴共点, 腕心 W 在 J4, 在 J6 的上游, 所以解算顺序是
        ``q6 -> W -> q1,q2 -> q3,q4,q5``, 与教科书球腕臂正好反过来:
          1. ``W56 = p + R·C_TCP`` (J5/J6 公垂线在 J6 轴上的点, 与 q6 无关);
          2. 腕心高度只归升降管 -> ``A sinθ6 + B cosθ6 = C``, 两个 θ6 根 (s6);
          3. ``W = W56 − d5·(sinθ6·x_tcp + cosθ6·y_tcp)``, 水平 2R 解 θ1, θ2 (s2, 肘内/外);
          4. ``N = Rx(−α2)·R02ᵀ·R05·Rx(π) = Rz(θ3)·Ry(−θ4)·Rz(−θ5)``, ZYZ 解 θ3..θ5 (s4, 翻腕)。
        推导与实测见 docs/2026-09-26-rm26-arm-wrist-structure-and-closed-form-ik.md 第四、九节。

        分支号 ``branch = 4·[s6<0] + 2·[s2<0] + [s4<0]``。 随机构型下真解主要落在 0 和 4
        (各约一半), 固定单分支会丢掉约一半目标。

        闭式解假设 ``d[3] = 0`` (J3/J4 两轴间 0.5 mm 的缝)。 ``d[3] != 0`` 时按分支做不动点修正
        (见 ``_correct_gap``), 再用 LM 在真 DH 上收尾; 因这 0.5 mm 刚好无解的目标, 高度 /
        水平 2R 夹到边界取最近点当初值。 ``valid`` = FK 复现目标 且 落在限位内。
        """
        targets = validate_transforms(target_tcp, name="target_tcp")
        branch_ids = self._normalize_branches(branches)
        batch_size = len(targets)
        theta3_ref = self._closed_form_theta3_reference(reference_joints, batch_size)
        solved, feasible = self._closed_form_candidates(targets, theta3_ref, singular_tolerance)
        solved = solved[:, branch_ids]
        feasible = feasible[:, branch_ids]
        repeated_targets = np.repeat(targets, len(branch_ids), axis=0)
        flat = solved.reshape(-1, 6)
        flat_feasible = feasible.reshape(-1)
        if abs(self.d[3]) > 0.0:
            flat, flat_feasible = self._correct_gap(
                flat,
                flat_feasible,
                repeated_targets,
                np.tile(np.asarray(branch_ids), batch_size),
                singular_tolerance,
            )
            flat = self._refine_ik(flat, repeated_targets, flat_feasible)
        joints, in_limits = self.joint_space.normalize(flat)
        reproduced = self._reproduces(
            joints, repeated_targets, position_tolerance, rotation_tolerance
        )
        valid = flat_feasible & in_limits & reproduced
        if abs(self.d[3]) > 0.0:
            joints, valid = self._jitter_fallback(
                joints.reshape(batch_size, len(branch_ids), 6),
                valid.reshape(batch_size, len(branch_ids)),
                targets,
                branch_ids,
                theta3_ref,
                singular_tolerance,
                position_tolerance,
                rotation_tolerance,
            )
        return joints.reshape(batch_size, len(branch_ids), 6), valid.reshape(batch_size, len(branch_ids))

    def _closed_form_candidates(
        self, targets: np.ndarray, theta3_ref: np.ndarray, singular_tolerance: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """d[3] 当 0 的闭式解, 返回全部 8 个分支的关节角 (B, 8, 6) 与几何可行 (B, 8)。"""
        batch_size = len(targets)
        rotations = targets[:, :3, :3]
        positions = targets[:, :3, 3]

        a1, l2, d5, d6 = self.a[1], self.d[2], self.d[4], self.d[5]
        c_tcp = np.asarray([0.0, 0.0, -d6]) - self.tool_offset
        wrist_height = self._closed_form_wrist_height()
        # d[3] 那条缝让腕心最多偏 |d[3]|: 高度方程 / 水平 2R 因此刚好越界的目标 (实测多是
        # 肘部接近伸直) 仍夹到边界给初值, 交给精修判真假。
        height_slack = 2.0 * abs(self.d[3]) + 1e-12
        reach_slack = 2.0 * (a1 + l2) * abs(self.d[3]) / (a1 * l2) + 1e-12

        # 第 1、2 步: W56 与 θ6 的两个根 (B, 2)
        w56 = positions + np.einsum("bij,j->bi", rotations, c_tcp)
        coef_a = d5 * rotations[:, 2, 0]
        coef_b = d5 * rotations[:, 2, 1]
        coef_c = w56[:, 2] - wrist_height
        radius = np.hypot(coef_a, coef_b)  # = d5·sin(β), β = J6 轴与竖直方向夹角
        height_ok = (radius > 1e-9) & (np.abs(coef_c) <= radius + height_slack)
        deviation = np.arccos(np.clip(coef_c / np.maximum(radius, 1e-12), -1.0, 1.0))
        theta6 = np.arctan2(coef_a, coef_b)[:, None] + np.asarray([1.0, -1.0]) * deviation[:, None]

        # 第 3 步: 腕心 + 水平 2R -> θ1, θ2 (B, 2, 2), 下标 [s6, s2]
        x_tcp = rotations[:, None, :, 0]
        y_tcp = rotations[:, None, :, 1]
        wrist = w56[:, None, :] - d5 * (
            np.sin(theta6)[..., None] * x_tcp + np.cos(theta6)[..., None] * y_tcp
        )
        planar = (wrist[..., 0] ** 2 + wrist[..., 1] ** 2 - a1 * a1 - l2 * l2) / (2.0 * a1 * l2)
        reach_ok = np.abs(planar) <= 1.0 + reach_slack
        base = np.arcsin(np.clip(planar, -1.0, 1.0))
        theta2 = np.stack((base, np.pi - base), axis=-1)
        theta1 = np.arctan2(wrist[..., 1], wrist[..., 0])[..., None] - np.arctan2(
            l2 * np.cos(theta2), a1 + l2 * np.sin(theta2)
        )

        # 第 4 步: 姿态 -> θ3, θ4, θ5 (B, 2, 2, 2), 下标 [s6, s2, s4]
        flat1 = theta1.reshape(-1)
        flat2 = theta2.reshape(-1)
        rotation_02 = (
            self._forward_dh_transforms(self.a[0], self.alpha[0], self.d[0], flat1)[:, :3, :3]
            @ self._forward_dh_transforms(self.a[1], self.alpha[1], self.d[1], flat2)[:, :3, :3]
        ).reshape(batch_size, 2, 2, 3, 3)
        rotation_05 = (
            rotations[:, None]
            @ _rot_z(-theta6.reshape(-1)).reshape(batch_size, 2, 3, 3)
            @ _rot_x(np.asarray([-self.alpha[5]]))[0]
        )
        wrist_zyz = (
            _rot_x(np.asarray([-self.alpha[2]]))[0]
            @ np.swapaxes(rotation_02, -1, -2)
            @ rotation_05[:, :, None]
            @ _rot_x(np.asarray([np.pi]))[0]
        )[:, :, :, None]  # (B, 2, 2, 1, 3, 3)
        t = np.asarray([1.0, -1.0]) * np.arccos(np.clip(wrist_zyz[..., 2, 2], -1.0, 1.0))
        sin_t = np.sin(t)
        regular = np.abs(sin_t) >= singular_tolerance
        safe = np.where(regular, sin_t, 1.0)
        theta3 = np.arctan2(wrist_zyz[..., 1, 2] / safe, wrist_zyz[..., 0, 2] / safe)
        theta4 = -t
        theta5 = -np.arctan2(wrist_zyz[..., 2, 1] / safe, -wrist_zyz[..., 2, 0] / safe)

        # 腕奇异: θ3 与 θ5 只能定和 (θ4=−π) / 差 (θ4=0)。 θ3 在"q3、q5 都落进限位"的
        # 区间里取离参考值最近的点, θ5 补出来 (参考值默认 q3 限位中点, 见 reference_joints)。
        theta3_ref = np.broadcast_to(theta3_ref[:, None, None, None], theta3.shape)
        upright = wrist_zyz[..., 2, 2] > 0.0
        singular_theta4 = np.where(upright, 0.0, -np.pi)
        # upright: θ5 − θ3 = −φ;  否则: θ3 + θ5 = φ'
        coupled = np.where(
            upright,
            -np.arctan2(wrist_zyz[..., 1, 0], wrist_zyz[..., 0, 0]),
            np.arctan2(-wrist_zyz[..., 1, 0], -wrist_zyz[..., 0, 0]),
        )
        theta3_singular = self._singular_theta3(theta3_ref, coupled, upright)
        singular_theta5 = np.where(upright, coupled + theta3_singular, coupled - theta3_singular)
        theta3 = np.where(regular, theta3, theta3_singular)
        theta4 = np.where(regular, theta4, singular_theta4)
        theta5 = np.where(regular, theta5, singular_theta5)

        shape = (batch_size, 2, 2, 2)
        physical = np.stack(
            (
                np.broadcast_to(theta1[..., None], shape),
                np.broadcast_to(theta2[..., None], shape),
                theta3,
                theta4,
                theta5,
                np.broadcast_to(theta6[:, :, None, None], shape),
            ),
            axis=-1,
        )
        solved = (physical - self.theta_offset).reshape(batch_size, 8, 6)
        feasible = np.broadcast_to(
            (height_ok[:, None] & reach_ok)[..., None, None], shape
        ).reshape(batch_size, 8)
        return solved, feasible

    def _correct_gap(
        self,
        joints: np.ndarray,
        feasible: np.ndarray,
        targets: np.ndarray,
        branch_ids: np.ndarray,
        singular_tolerance: float,
        *,
        max_iterations: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """补 d[3] 那 0.5 mm: 不动点迭代 ``q <- IK0_b(T · FK(q)⁻¹ · FK0(q))``。

        FK0 是 d[3]=0 的模型, ``FK(q)⁻¹·FK0(q)`` 是亚毫米的修正量, 不动点满足 ``FK(q) = T``。
        每步都是同一分支的精确闭式解, 所以不会像纯局部优化那样在奇异附近跳到别的分支
        (实测纯 LM 约 2% 目标收敛到别的解或卡住)。 奇异时 θ3 参考值取上一步的 q3 保持连续。
        """
        gap_free = replace(self, d=np.where(np.arange(6) == 3, 0.0, self.d))
        joints = joints.copy()
        feasible = feasible.copy()
        # 腕奇异附近映射会在两点间来回跳 (周期 2), 残差不降就把该行步长减半
        relax = np.ones(len(joints))
        previous = np.full(len(joints), np.inf)
        active = feasible.copy()
        for _ in range(max_iterations):
            index = np.flatnonzero(active)
            if not len(index):
                break
            current = joints[index]
            tcp = self.forward_kinematics(current).tcp_transforms
            error = np.linalg.norm(tcp[:, :3, 3] - targets[index, :3, 3], axis=1) + np.linalg.norm(
                tcp[:, :3, :3] - targets[index, :3, :3], axis=(1, 2)
            )
            active[index[error < 1e-13]] = False
            relax[index] = np.where(
                error < previous[index], relax[index], np.maximum(relax[index] * 0.5, 0.125)
            )
            previous[index] = np.minimum(error, previous[index])
            inverse = np.zeros_like(tcp)
            inverse[:, :3, :3] = np.swapaxes(tcp[:, :3, :3], 1, 2)
            inverse[:, :3, 3] = -np.einsum("bij,bj->bi", inverse[:, :3, :3], tcp[:, :3, 3])
            inverse[:, 3, 3] = 1.0
            corrected = targets[index] @ inverse @ gap_free.forward_kinematics(current).tcp_transforms
            candidates, candidate_feasible = self._closed_form_candidates(
                corrected, current[:, 2] + self.theta_offset[2], singular_tolerance
            )
            rows = np.arange(len(index))
            step = candidates[rows, branch_ids[index]] - current
            step = (step + np.pi) % (2.0 * np.pi) - np.pi
            joints[index] = current + relax[index, None] * step
            feasible[index] = candidate_feasible[rows, branch_ids[index]]
            active &= feasible
        return joints, feasible

    def _jitter_fallback(
        self,
        joints: np.ndarray,
        valid: np.ndarray,
        targets: np.ndarray,
        branch_ids: tuple[int, ...],
        theta3_ref: np.ndarray,
        singular_tolerance: float,
        position_tolerance: float,
        rotation_tolerance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """一支都没解出的目标 (约 0.4%, 多在腕奇异 / 高度边界附近): 目标 z 在 ±1.2·|d[3]| 内
        抖 13 个点取闭式初值, 再对原目标修正 + 精修 (文档第九节)。 只补 ``valid`` 为假的目标,
        分支号仍按初值所在分支记, 精修后的解可能与名义分支不同, 所以 FK 复现与限位照常检查。"""
        failed = np.flatnonzero(~valid.any(axis=1))
        if not len(failed):
            return joints, valid
        offsets = np.linspace(-1.2, 1.2, 13) * abs(self.d[3])
        count, width = len(failed), len(branch_ids)
        shifted = np.repeat(targets[failed], len(offsets), axis=0)
        shifted[:, 2, 3] += np.tile(offsets, count)
        seeds, seed_feasible = self._closed_form_candidates(
            shifted, np.repeat(theta3_ref[failed], len(offsets)), singular_tolerance
        )
        seeds = seeds[:, branch_ids].reshape(-1, 6)
        seed_feasible = seed_feasible[:, branch_ids].reshape(-1)
        original = np.repeat(targets[failed], len(offsets) * width, axis=0)
        branch_index = np.tile(np.asarray(branch_ids), count * len(offsets))
        seeds, seed_feasible = self._correct_gap(
            seeds, seed_feasible, original, branch_index, singular_tolerance
        )
        seeds = self._refine_ik(seeds, original, seed_feasible)
        normalized, in_limits = self.joint_space.normalize(seeds)
        ok = seed_feasible & in_limits & self._reproduces(normalized, original, position_tolerance, rotation_tolerance)
        normalized = normalized.reshape(count, len(offsets), width, 6)
        ok = ok.reshape(count, len(offsets), width)
        pick = np.argmax(ok, axis=1)  # 每个 (目标, 分支) 取第一个成功的抖动点
        rows = np.arange(count)[:, None]
        cols = np.arange(width)[None, :]
        joints = joints.copy()
        valid = valid.copy()
        joints[failed] = normalized[rows, pick, cols]
        valid[failed] = ok[rows, pick, cols]
        return joints, valid

    def _closed_form_wrist_height(self) -> float:
        """腕心在 arm_base 系的高度 (d[3] 当 0), 与关节角无关; 从 FK 取, 不手推号。"""
        zeros = np.zeros(1)
        transform = np.eye(4)
        for index in range(4):
            d = 0.0 if index == 3 else self.d[index]
            transform = transform @ self._forward_dh_transforms(
                self.a[index], self.alpha[index], d, zeros + self.theta_offset[index]
            )[0]
        return float(transform[2, 3])

    def _singular_theta3(
        self, theta3_ref: np.ndarray, coupled: np.ndarray, upright: np.ndarray
    ) -> np.ndarray:
        """腕奇异时挑 θ3: q5 = ±q3 + 常数 (upright 取 +, 否则取 −), 求使 q3、q5 同时在限位内的
        q3 区间 (考虑 2π 代表), 返回其中离参考值最近的点; 区间为空就原样返回参考值。"""
        lower3, upper3 = self.joint_space.lower[2], self.joint_space.upper[2]
        lower5, upper5 = self.joint_space.lower[4], self.joint_space.upper[4]
        if not all(np.isfinite((lower3, upper3, lower5, upper5))):
            return theta3_ref
        offset3, offset5 = self.theta_offset[2], self.theta_offset[4]
        q3_ref = theta3_ref - offset3
        # upright: q5 = q3 + k,  k = coupled + offset3 − offset5  -> q3 ∈ [lower5 − k, upper5 − k]
        # 否则:    q5 = k − q3,  k = coupled − offset3 − offset5  -> q3 ∈ [k − upper5, k − lower5]
        k = np.where(upright, coupled + offset3 - offset5, coupled - offset3 - offset5)
        low = np.where(upright, lower5 - k, k - upper5)
        high = np.where(upright, upper5 - k, k - lower5)
        best = np.full(q3_ref.shape, np.nan)
        best_distance = np.full(q3_ref.shape, np.inf)
        for turns in range(-3, 4):
            # q5 的 2π 代表平移 q3 区间; q3 自己再按 2π 代表对齐参考值
            shift = 2.0 * np.pi * turns
            lo_i = np.maximum(low + shift, lower3)
            hi_i = np.minimum(high + shift, upper3)
            candidate = np.clip(q3_ref, lo_i, hi_i)
            distance = np.abs(candidate - q3_ref)
            better = (lo_i <= hi_i) & (distance < best_distance)
            best = np.where(better, candidate, best)
            best_distance = np.where(better, distance, best_distance)
        return np.where(np.isfinite(best), best + offset3, theta3_ref)

    def _closed_form_theta3_reference(self, reference_joints, batch_size: int) -> np.ndarray:
        if reference_joints is None:
            lower, upper = self.joint_space.lower[2], self.joint_space.upper[2]
            q3 = 0.5 * (lower + upper) if np.isfinite(lower) and np.isfinite(upper) else 0.0
            return np.full(batch_size, q3 + self.theta_offset[2])
        reference = np.asarray(reference_joints, dtype=float)
        if reference.shape == (6,):
            reference = np.broadcast_to(reference, (batch_size, 6))
        if reference.shape != (batch_size, 6):
            raise ValueError(
                f"reference_joints must have shape (6,) or ({batch_size}, 6), got {reference.shape}"
            )
        return reference[:, 2] + self.theta_offset[2]

    def _refine_ik(
        self,
        seeds: np.ndarray,
        targets: np.ndarray,
        active: np.ndarray,
        *,
        max_iterations: int = 50,
        tolerance: float = 1e-12,
    ) -> np.ndarray:
        """批量 Levenberg-Marquardt: 以闭式解为初值在真 DH 上把 FK 残差压到 0。

        残差 = [Δp (m), 旋转误差的 rotvec (rad)], 雅可比用几何雅可比 (Craig DH 第 i 轴
        是 link_transforms[i+1] 的 z 轴)。 初值离真解通常只有亚毫米, 但目标附近常接近奇异
        (腕心高度几乎只归升降管), 纯高斯-牛顿会过冲发散, 所以每行各自按代价升降阻尼。
        """
        joints = seeds.copy()
        active = active.copy()
        damping = np.full(len(joints), 1e-6)

        def evaluate(q: np.ndarray, target: np.ndarray):
            fk = self.forward_kinematics(q)
            tcp = fk.tcp_transforms
            residual = np.concatenate(
                (
                    tcp[:, :3, 3] - target[:, :3, 3],
                    Rotation.from_matrix(
                        tcp[:, :3, :3] @ np.swapaxes(target[:, :3, :3], 1, 2)
                    ).as_rotvec(),
                ),
                axis=1,
            )
            return fk, residual

        for _ in range(max_iterations):
            index = np.flatnonzero(active)
            if not len(index):
                break
            fk, residual = evaluate(joints[index], targets[index])
            cost = np.sum(residual * residual, axis=1)
            converged = cost < tolerance * tolerance
            active[index[converged]] = False
            keep = ~converged
            if not np.any(keep):
                break
            index, residual, cost = index[keep], residual[keep], cost[keep]
            axes = fk.link_transforms[keep, 1:, :3, 2]
            lever = fk.tcp_transforms[keep, None, :3, 3] - fk.link_transforms[keep, 1:, :3, 3]
            jacobian = np.concatenate((np.cross(axes, lever), axes), axis=2).transpose(0, 2, 1)
            normal = np.swapaxes(jacobian, 1, 2) @ jacobian
            gradient = np.einsum("bji,bj->bi", jacobian, residual)
            diagonal = np.einsum("bii->bi", normal)
            normal = normal + (damping[index, None] * np.maximum(diagonal, 1e-9))[..., None] * np.eye(6)
            candidate = joints[index] - np.linalg.solve(normal, gradient[..., None])[..., 0]
            _, candidate_residual = evaluate(candidate, targets[index])
            improved = np.sum(candidate_residual * candidate_residual, axis=1) < cost
            joints[index[improved]] = candidate[improved]
            damping[index] = np.where(
                improved, np.maximum(damping[index] / 3.0, 1e-12), damping[index] * 4.0
            )
            active[index[damping[index] > 1e8]] = False
        return joints

    def _reproduces(
        self,
        joints: np.ndarray,
        targets: np.ndarray,
        position_tolerance: float,
        rotation_tolerance: float,
    ) -> np.ndarray:
        tcp = self.forward_kinematics(joints).tcp_transforms
        position_error = np.linalg.norm(tcp[:, :3, 3] - targets[:, :3, 3], axis=1)
        rotation_error = np.linalg.norm(tcp[:, :3, :3] - targets[:, :3, :3], axis=(1, 2))
        return (position_error < position_tolerance) & (rotation_error < rotation_tolerance)

    def _solve_ik_numeric(
        self,
        target_tcp: np.ndarray,
        branches: tuple[int, ...] | list[int] | None = None,
        *,
        position_tolerance: float = 1e-4,
        rotation_tolerance: float = 1e-3,
        max_nfev: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """数值 IK 兜底: 对每个目标位姿用 scipy least_squares 最小化 FK 残差 (位置+姿态)。

        与臂型无关, 只依赖 FK, 对偏置腕/球腕都适用 (见 README 6.3 迁移路线)。 保持与
        解析 IK 相同的返回契约: 请求的每个 branch 用一个不同初值求解 (类比解析法的多分支),
        branch 0 从关节中点起步, 其余用确定性随机初值, 以便在多解里撒开。 ``valid`` 为
        "解出的位姿确实复现了目标 且 落在限位内"。
        """
        targets = validate_transforms(target_tcp, name="target_tcp")
        branch_ids = self._normalize_branches(branches)
        joints = np.zeros((len(targets), len(branch_ids), 6), dtype=float)
        valid = np.zeros((len(targets), len(branch_ids)), dtype=bool)

        lower, upper = self.joint_space.lower, self.joint_space.upper
        finite_lower = np.where(np.isfinite(lower), lower, -np.pi)
        finite_upper = np.where(np.isfinite(upper), upper, np.pi)
        midpoint = 0.5 * (finite_lower + finite_upper)

        def residual(q: np.ndarray, target: np.ndarray) -> np.ndarray:
            tcp = self.forward_kinematics(np.asarray(q)[None, :]).tcp_transforms[0]
            rotation_error = Rotation.from_matrix(target[:3, :3].T @ tcp[:3, :3]).as_rotvec()
            return np.concatenate((tcp[:3, 3] - target[:3, 3], rotation_error))

        for batch_index, target in enumerate(targets):
            for output_index, branch in enumerate(branch_ids):
                seed = midpoint if branch == 0 else np.random.default_rng(branch).uniform(
                    finite_lower, finite_upper
                )
                solution = least_squares(
                    residual, seed, args=(target,), bounds=(lower, upper), method="trf",
                    x_scale="jac", max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10,
                )
                error = residual(solution.x, target)
                reproduced = (
                    np.linalg.norm(error[:3]) < position_tolerance
                    and np.linalg.norm(error[3:]) < rotation_tolerance
                )
                normalized, in_limits = self.joint_space.normalize(np.asarray(solution.x)[None, :])
                joints[batch_index, output_index] = normalized[0]
                valid[batch_index, output_index] = bool(reproduced and in_limits[0])
        return joints, valid

    def forward_kinematics(self, joints: np.ndarray) -> ForwardKinematics:
        values = self._validate_joints(joints)
        physical = values + self.theta_offset[None, :]
        batch_size = len(values)
        links = np.broadcast_to(np.eye(4), (batch_size, 7, 4, 4)).copy()
        cumulative = links[:, 0].copy()
        for index in range(6):
            cumulative = cumulative @ self._forward_dh_transforms(
                self.a[index], self.alpha[index], self.d[index], physical[:, index]
            )
            links[:, index + 1] = cumulative
        tcp = links[:, -1].copy()
        tcp[:, :3, 3] += np.einsum("bij,j->bi", tcp[:, :3, :3], self.tool_offset)
        return ForwardKinematics(link_transforms=links, tcp_transforms=tcp)

    def inverse_dynamics(
        self,
        joints: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        *,
        chassis_acceleration: np.ndarray | None = None,
        include_link_inertia: bool = False,
    ) -> np.ndarray:
        q = self._validate_joints(joints)
        dq = self._validate_joints(velocities, name="velocities")
        ddq = self._validate_joints(accelerations, name="accelerations")
        if dq.shape != q.shape or ddq.shape != q.shape:
            raise ValueError("joints, velocities, and accelerations must have identical shapes")
        chassis = np.zeros(len(q)) if chassis_acceleration is None else np.asarray(chassis_acceleration, dtype=float)
        if chassis.shape != (len(q),):
            raise ValueError(f"chassis_acceleration must have shape ({len(q)},), got {chassis.shape}")
        forward, inverse = self._dh_rotation_chains(q)
        translations = self._parent_translations()
        axis = np.asarray([0.0, 0.0, 1.0])
        angular_velocity = np.zeros((len(q), 7, 3))
        angular_acceleration = np.zeros((len(q), 7, 3))
        linear_acceleration = np.zeros((len(q), 7, 3))
        linear_acceleration[:, 0, 1] = -chassis
        linear_acceleration[:, 0, 2] = 9.81
        forces = np.zeros((len(q), 6, 3))
        moments = np.zeros((len(q), 6, 3))

        for index in range(6):
            rotated_velocity = np.einsum(
                "bij,bj->bi", inverse[:, index], angular_velocity[:, index]
            )
            joint_velocity = dq[:, index, None] * axis
            angular_velocity[:, index + 1] = rotated_velocity + joint_velocity
            angular_acceleration[:, index + 1] = (
                np.einsum(
                    "bij,bj->bi",
                    inverse[:, index],
                    angular_acceleration[:, index],
                )
                + np.cross(rotated_velocity, joint_velocity)
                + ddq[:, index, None] * axis
            )
            parent_acceleration = (
                np.cross(angular_acceleration[:, index], translations[index])
                + np.cross(
                    angular_velocity[:, index],
                    np.cross(angular_velocity[:, index], translations[index]),
                )
                + linear_acceleration[:, index]
            )
            linear_acceleration[:, index + 1] = np.einsum(
                "bij,bj->bi", inverse[:, index], parent_acceleration
            )
            center = self.centers_of_mass[index]
            center_acceleration = (
                np.cross(angular_acceleration[:, index + 1], center)
                + np.cross(
                    angular_velocity[:, index + 1],
                    np.cross(angular_velocity[:, index + 1], center),
                )
                + linear_acceleration[:, index + 1]
            )
            forces[:, index] = self.masses[index] * center_acceleration
            inertia_velocity = np.einsum(
                "ij,bj->bi", self.inertias[index], angular_velocity[:, index + 1]
            )
            moments[:, index] = np.einsum(
                "ij,bj->bi", self.inertias[index], angular_acceleration[:, index + 1]
            ) + np.cross(angular_velocity[:, index + 1], inertia_velocity)

        return self._rnea_backward(
            forward,
            forces,
            moments,
            np.zeros((len(q), 3)),
            np.zeros((len(q), 3)),
            include_link_inertia,
        )

    def external_wrench_torque(self, joints: np.ndarray, tcp_wrenches: np.ndarray) -> np.ndarray:
        q = self._validate_joints(joints)
        wrenches = np.asarray(tcp_wrenches, dtype=float)
        if wrenches.shape != (len(q), 6):
            raise ValueError(f"tcp_wrenches must have shape ({len(q)}, 6), got {wrenches.shape}")
        force = wrenches[:, :3]
        torque_frame6 = wrenches[:, 3:] + np.cross(self.tool_offset, force)
        forward, _ = self._dh_rotation_chains(q)
        zeros = np.zeros((len(q), 6, 3))
        return self._rnea_backward(forward, zeros, zeros, force, torque_frame6, False)

    def _validate_joints(self, joints: np.ndarray, *, name: str = "joints") -> np.ndarray:
        values = np.asarray(joints, dtype=float)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(f"{name} must have shape (B, 6), got {values.shape}")
        return values

    @staticmethod
    def _normalize_branches(branches) -> tuple[int, ...]:
        values = tuple(range(8)) if branches is None else tuple(int(value) for value in branches)
        if not values or len(set(values)) != len(values) or any(value < 0 or value >= 8 for value in values):
            raise ValueError(f"IK branches must be unique values in [0, 7], got {values}")
        return values

    @staticmethod
    def _inverse_dh_rotations(theta: np.ndarray, alpha: float) -> np.ndarray:
        cosine = np.cos(theta)
        sine = np.sin(theta)
        cosine_alpha = np.cos(alpha)
        sine_alpha = np.sin(alpha)
        result = np.empty((len(theta), 3, 3), dtype=float)
        result[:, 0, 0] = cosine
        result[:, 0, 1] = sine * cosine_alpha
        result[:, 0, 2] = sine * sine_alpha
        result[:, 1, 0] = -sine
        result[:, 1, 1] = cosine * cosine_alpha
        result[:, 1, 2] = cosine * sine_alpha
        result[:, 2, 0] = 0.0
        result[:, 2, 1] = -sine_alpha
        result[:, 2, 2] = cosine_alpha
        return result

    @staticmethod
    def _forward_dh_transforms(a: float, alpha: float, d: float, theta: np.ndarray) -> np.ndarray:
        cosine = np.cos(theta)
        sine = np.sin(theta)
        cosine_alpha = np.cos(alpha)
        sine_alpha = np.sin(alpha)
        result = np.zeros((len(theta), 4, 4), dtype=float)
        result[:, 0, 0] = cosine
        result[:, 0, 1] = -sine
        result[:, 0, 3] = a
        result[:, 1, 0] = sine * cosine_alpha
        result[:, 1, 1] = cosine * cosine_alpha
        result[:, 1, 2] = -sine_alpha
        result[:, 1, 3] = -d * sine_alpha
        result[:, 2, 0] = sine * sine_alpha
        result[:, 2, 1] = cosine * sine_alpha
        result[:, 2, 2] = cosine_alpha
        result[:, 2, 3] = d * cosine_alpha
        result[:, 3, 3] = 1.0
        return result

    def _dh_rotation_chains(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self._validate_joints(joints)
        forward = np.broadcast_to(np.eye(3), (len(values), 7, 3, 3)).copy()
        inverse = np.empty((len(values), 6, 3, 3))
        for index in range(6):
            theta = values[:, index] + self.theta_offset[index]
            transform = self._forward_dh_transforms(
                self.a[index], self.alpha[index], self.d[index], theta
            )
            forward[:, index] = transform[:, :3, :3]
            inverse[:, index] = np.swapaxes(transform[:, :3, :3], 1, 2)
        return forward, inverse

    def _parent_translations(self) -> np.ndarray:
        return np.stack(
            (
                self.a,
                -self.d * np.sin(self.alpha),
                self.d * np.cos(self.alpha),
            ),
            axis=1,
        )

    def _rnea_backward(self, forward, forces, moments, tip_force, tip_torque, include_link_inertia):
        translations = np.vstack((self._parent_translations(), np.zeros(3)))
        batch_size = len(forward)
        force = np.zeros((batch_size, 7, 3))
        torque = np.zeros((batch_size, 7, 3))
        force[:, 6] = np.asarray(tip_force, dtype=float)
        torque[:, 6] = np.asarray(tip_torque, dtype=float)
        joint_torque = np.zeros((batch_size, 6))
        for frame in range(6, 0, -1):
            propagated_force = np.einsum(
                "bij,bj->bi", forward[:, frame], force[:, frame]
            )
            force[:, frame - 1] = propagated_force + forces[:, frame - 1]
            torque[:, frame - 1] = (
                np.einsum("bij,bj->bi", forward[:, frame], torque[:, frame])
                + np.cross(translations[frame], propagated_force)
                + np.cross(self.centers_of_mass[frame - 1], forces[:, frame - 1])
            )
            if include_link_inertia:
                torque[:, frame - 1] += moments[:, frame - 1]
            joint_torque[:, frame - 1] = torque[:, frame - 1, 2]
        return joint_torque


def _rot_x(theta: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(theta), np.sin(theta)
    result = np.zeros((len(theta), 3, 3))
    result[:, 0, 0] = 1.0
    result[:, 1, 1] = cosine
    result[:, 1, 2] = -sine
    result[:, 2, 1] = sine
    result[:, 2, 2] = cosine
    return result


def _rot_z(theta: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(theta), np.sin(theta)
    result = np.zeros((len(theta), 3, 3))
    result[:, 0, 0] = cosine
    result[:, 0, 1] = -sine
    result[:, 1, 0] = sine
    result[:, 1, 1] = cosine
    result[:, 2, 2] = 1.0
    return result

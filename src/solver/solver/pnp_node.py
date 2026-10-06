"""ROS 薄节点：将 detector 观测转换为相机系模型位姿。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import load_schema
from .solver import PnPEstimator


def _camera_info_to_arrays(msg):
    K = np.asarray(msg.k, dtype=np.float64)
    if K.size != 9:
        raise ValueError("CameraInfo K must contain 9 values")
    K = K.reshape(3, 3)
    D = np.asarray(msg.d, dtype=np.float64).reshape(-1)
    if D.size not in (0, 4, 5, 8, 12, 14):
        raise ValueError("CameraInfo D has invalid length")
    if not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError("CameraInfo K/D must be finite")
    return K, D


def _observation_to_arrays(msg):
    u = np.asarray(msg.u, dtype=np.float64).reshape(-1)
    v = np.asarray(msg.v, dtype=np.float64).reshape(-1)
    confidence = np.asarray(msg.confidence, dtype=np.float64).reshape(-1)
    if not (len(u) == len(v) == len(confidence)):
        raise ValueError("observation parallel arrays must have equal length")
    points = np.column_stack((u, v))
    if not np.isfinite(points).all() or not np.isfinite(confidence).all():
        raise ValueError("observation arrays must be finite")
    return str(msg.schema), int(msg.class_id), points, confidence


def _config_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir() and (parent / "src").is_dir():
            return parent / "config"
    return here.parents[3] / "config"


def _stamp_to_msg(stamp):
    return stamp


class PnpNode:
    def __init__(self):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from interfaces.msg import KeypointObservation
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo

        class _Node(Node):
            def __init__(self, outer):
                super().__init__("pnp_node")
                self.outer = outer
                self.declare_parameter("observation_topic", "/detector/keypoints")
                self.declare_parameter("camera_info_topic", "/camera/camera_info")
                self.declare_parameter("output_topic", "/solver/camera_station_model_pose")
                self.declare_parameter("confidence_threshold", 0.2)
                self.declare_parameter("reprojection_error_threshold_px", 5.0)
                self._camera_info = None
                self._schemas = {}
                self._estimator = PnPEstimator(
                    self.get_parameter("confidence_threshold").value,
                    self.get_parameter("reprojection_error_threshold_px").value,
                )
                self._pub = self.create_publisher(
                    PoseStamped, self.get_parameter("output_topic").value, 10
                )
                self.create_subscription(CameraInfo, self.get_parameter("camera_info_topic").value,
                                         self._on_camera_info, qos_profile_sensor_data)
                self.create_subscription(KeypointObservation,
                                         self.get_parameter("observation_topic").value,
                                         self._on_observation, qos_profile_sensor_data)

            def _on_camera_info(self, msg):
                self._camera_info = msg

            def _on_observation(self, msg):
                if self._camera_info is None:
                    self.get_logger().warning("等待 CameraInfo，跳过观测")
                    return
                if msg.header.frame_id and self._camera_info.header.frame_id and msg.header.frame_id != self._camera_info.header.frame_id:
                    self.get_logger().warning("观测与 CameraInfo frame_id 不一致")
                    return
                try:
                    schema_name, class_id, image_points, confidence = _observation_to_arrays(msg)
                    schema = self._schemas.get(schema_name)
                    if schema is None:
                        schema = load_schema(_config_dir() / "keypoint_schema.yaml", schema_name)
                        self._schemas[schema_name] = schema
                    if schema.class_id != class_id:
                        raise ValueError(f"class_id {class_id} does not match schema {schema_name}")
                    K, D = _camera_info_to_arrays(self._camera_info)
                    result = self._estimator.estimate(
                        schema.points_for_observation(len(image_points)), image_points, K, D, confidence
                    )
                except (KeyError, ValueError) as exc:
                    self.get_logger().warning(f"无效视觉观测: {exc}")
                    return
                if not result.valid:
                    self.get_logger().warning(f"PnP 无效: {result.reason}")
                    return
                pose = PoseStamped()
                pose.header = self._camera_info.header
                pose.header.stamp = msg.header.stamp
                pose.header.frame_id = self._camera_info.header.frame_id
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = result.T_camera_station_model[:3, 3]
                qw, qx, qy, qz = _rotation_to_quaternion(result.T_camera_station_model[:3, :3])
                pose.pose.orientation.w, pose.pose.orientation.x = qw, qx
                pose.pose.orientation.y, pose.pose.orientation.z = qy, qz
                self._pub.publish(pose)

        self._node = _Node(self)

    def spin(self):
        import rclpy
        rclpy.spin(self._node)


def _rotation_to_quaternion(rotation):
    import cv2
    rotvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    axis = rotvec[:, 0] / theta
    half = theta / 2.0
    return float(np.cos(half)), *(float(x) for x in axis * np.sin(half))


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = PnpNode()
    try:
        node.spin()
    finally:
        node._node.destroy_node()
        rclpy.shutdown()

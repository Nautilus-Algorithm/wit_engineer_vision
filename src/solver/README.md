# solver 包说明

## 1. 包定位

`solver` 是视觉位姿解算包，目标是把 detector 输出的二维关键点与目标三维模型点、相机内参结合，求出兑换站在相机光学坐标系中的 6D 位姿。

```text
KeypointObservation + CameraInfo + keypoint_schema
    → T_camera_station_model
```

本包不是机械臂逆运动学包。当前机器人软件接口严格使用单臂六轴 J1–J6；关节限位、FK、IK、碰撞和轨迹均属于 `planning`。底盘轮组和升降轴 `line` 不属于本包接口。

## 2. 当前状态

纯算法层和 ROS PnP 节点已实现：

- `schema.py` 校验根目录 schema，并保持 YAML 关键点顺序；
- `solver.py` 提供 NumPy/OpenCV PnP、正深度、内点和重投影误差检查；
- `pnp_node.py` 订阅 `/detector/keypoints` 与 `/camera/camera_info`，发布相机系 `PoseStamped` 到 `/solver/camera_station_model_pose`；
- `config/keypoint_schema.yaml` 的 3D 坐标仍为 TODO，因此真实节点在填入实测坐标前不会产生有效位姿。

当前业务边界是**单臂、单个兑换站目标、单次目标执行**。solver 每次只处理一个有效兑换站观测，不实现多个兑换柱/兑换口的实例配对、目标选择、调度或多机械臂分配。`ExchangeStationPose.msg` 中的 `station_id` 暂时保留用于消息兼容性，不代表当前支持多站点业务。

因此真实视觉链路尚未打通。当前仿真 `/vision/exchange_pose` 由 `sim_node` 提供真值，不能作为 solver 已经可用的证明。

## 3. 与其它包的边界

```text
camera_node
  ├─ /camera/image_raw
  └─ /camera/camera_info
          ↓
detector_node
  └─ /detector/keypoints : interfaces/KeypointObservation
          ↓
pnp_node（本包）
  └─ T_camera_station_model
          ↓
tf / vision_tf_node
  └─ /vision/exchange_pose : interfaces/ExchangeStationPose
          ↓
planning_node
  └─ 六轴 FK / IK / 轨迹 / 碰撞
```

### 本包负责

- 读取并校验单个兑换站目标的 3D schema；
- 将该目标的二维关键点按固定 schema 顺序与三维点对应；
- 调用 OpenCV PnP；
- 计算重投影误差、内点和深度质量；
- 在 ROS 节点层完成消息转换、时间戳和诊断。

### 本包不负责

- 目标检测或关键点推理；
- camera→arm_base 手眼变换；
- `station_model→station_assembly` 固定变换；
- 机械臂 IK、关节选择或轨迹规划；
- 读取关节状态并生成关节角。

算法模块禁止 `import rclpy`，也禁止导入 `interfaces.msg`。ROS 消息只在 `pnp_node.py` 中转换为 NumPy 数组。

## 4. 坐标系与输出语义

### 4.1 `station_model`

`config/keypoint_schema.yaml` 中的三维点属于目标模型坐标系，建议统一称为 `station_model`：

- 原点：兑换口中心；
- X：右；
- Y：下；
- Z：朝外；
- 单位：米。

PnP 输出的矩阵应命名为 `T_camera_station_model`，含义是把模型系点变换到 `camera_optical`：

```text
p_camera = T_camera_station_model · p_station_model
```

### 4.2 不要混用装配系

sim 中由 `station.from_joint` 构造的姿态对应插到底 TCP 的装配基准，约定沿 +x 退回到接近位置。它应称为 `station_assembly`，不是 `station_model`。

真实视觉链需要明确的固定变换：

```text
T_arm_base_station_assembly
  = T_arm_base_camera
  · T_camera_station_model
  · T_station_model_station_assembly
```

`T_station_model_station_assembly` 尚未有真实值。solver 不得跳过它，也不得把 `T_camera_station_model` 直接发布成 planning 可执行的装配目标。

`ExchangeStationPose.msg` 当前使用 `header.frame_id = arm_base`，其 `pose` 应约定表示 `arm_base → station_assembly`。如果未来需要同时发布 model 和 assembly 两种位姿，必须先修改 `.msg` 并同步更新 `README_PRE.md`，不能靠隐式命名区分。

## 5. 推荐模块划分

```text
solver/
├── schema.py    # schema 读取、点名与单位校验
├── solver.py    # 纯 NumPy/OpenCV PnP 与质量结果
├── filter.py    # 可选时间滤波，不保存安全执行状态
└── pnp_node.py  # ROS 订阅、消息转换、参数和诊断
```

建议纯算法公开接口类似：

```python
result = estimator.estimate(
    object_points_m,       # (N, 3), station_model
    image_points_px,       # (N, 2), camera_optical
    camera_matrix,         # (3, 3)
    distortion_coeffs,     # (D,)
)
```

结果至少应包含：

- `T_camera_station_model`，形状 `(4, 4)`；
- `reprojection_error_px`；
- `(N,)` 的内点掩码；
- `valid` 和可诊断失败原因。

核心 API 不应返回 ROS 消息。

## 6. PnP 实现建议

参考 `~/RM2026-Engineer-Assembly-Algorithm` 的 `perception/pose.py`，推荐流程：

1. 根据 `schema` 获取 3D 点；
2. 校验 2D/3D/confidence 数量一致且均为有限值；
3. 过滤低置信度点，至少保留 4 对非退化点；
4. 使用 `solvePnPRansac`（优先 SQPNP）获得鲁棒初值；
5. 用 `solvePnPRefineLM` 精化候选解；
6. 检查所有有效点变换后的深度为正；
7. 计算全部有效点的重投影误差；
8. 根据内点数、误差、有限性和深度检查设置 `valid`；
9. 以明确的父子 frame 语义交给后续 tf 层。

重投影误差单位是像素，不能当作机械臂末端的米制误差。阈值必须配置化并通过实际相机、关键点精度和装配间隙验证。

## 7. 输入契约

`KeypointObservation.msg` 当前字段为：

- `schema`：例如 `exchange`；
- `class_id`；
- `u[]` / `v[]`：像素坐标；
- `confidence[]`：逐点置信度。

消息不携带关键点名称，因此数组顺序必须严格对应 detector 配置和 schema。当前 `tc` 关键点顺序由 `config/detector.yaml` 给出；修改权重或顺序后必须同步核对，错误顺序可能产生“收敛但错误”的 PnP 结果。

相机内参来自 `/camera/camera_info`，并应检查：

- `K` 为 `(3, 3)`；
- `D` 长度合法；
- `CameraInfo.header.frame_id` 与图像 optical frame 一致；
- 时间戳和图像匹配到可接受范围。

## 8. 配置

| 配置 | 用途 | 当前状态 |
|---|---|---|
| `config/keypoint_schema.yaml` | 目标名、class_id、3D 点 | 12 点坐标仍 TODO |
| `config/detector.yaml` | detector 后端和关键点顺序 | 已有 tc 顺序；默认后端为 eu |
| `config/camera_info.yaml` | 相机内参文件 | 已有标定数值，但运行时优先使用 CameraInfo |
| `config/vision_tf.yaml` | camera→arm_base 外参 | 单位阵占位，由 tf 使用 |
| `config/planning.yaml` | DH、限位、工具偏置 | solver 不读取 |

不要在包内复制这些 YAML。所有共享真值位于仓库根 `config/`。

## 9. 错误处理

以下情况必须返回无效结果或拒绝发布可执行位姿：

- schema 不存在；
- 关键点数量、顺序或数组长度不一致；
- 有效点少于 4 个或几何退化；
- 相机矩阵错误；
- PnP 不收敛；
- 深度非正；
- 重投影误差超过阈值；
- 图像和 CameraInfo frame 不一致；
- 外参或 model→assembly 固定变换仍为占位。

可以保留上一帧用于调试画面，但不得静默把上一帧作为新的安全执行目标。无效状态不能通过发送零位姿表示；应显式使用 `valid=false` 和诊断信息。

## 10. 六轴适配原则

当前 J1–J6 限位为：

```text
J1 [-0.52, 1.91]   J2 [-0.52, 2.61]   J3 [-3.14, 3.14]
J4 [ 0.00, 1.57]   J5 [-1.39, 4.36]   J6 [-1.57, 1.57]  rad
```

这些值属于 planning，不能复制到 solver。solver 只需输出正确的目标位姿；可达性、IK 分支连续性、碰撞和轨迹限速由 planning 检查。

禁止：

- 在 solver 中加入 IK；
- 将位姿的 6 个分量当成 6 个关节角；
- 把 `line` 或轮组加入消息；
- 让视觉解算依赖 `rm26_arm` / `spherical_wrist` 等 IK 选项。

## 11. 测试与落地顺序

测试命令：

```bash
cd src/solver
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD python3 -m pytest test -q
```

其中确定性仿真测试不依赖相机、模型权重或显示器；真实相机测试默认跳过，显式运行：

```bash
RUN_REAL_CAMERA_TEST=1 REAL_DETECTOR_MODEL=/path/to/openvino.xml \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD python3 -m pytest test/test_real_camera_detector_solver.py -q
```

启用真实测试后，缺少 `ros2`、模型、相机启动失败、超时或没有完整检测数据流都会使测试失败，而不是静默跳过。
## 12. 实现前置条件

按顺序处理：

1. 根据官方图纸/实测填写 3D schema；
2. 固化 detector 关键点名称与顺序并增加启动自检；
3. 实现并测试纯 PnP 算法；
4. 实现 `pnp_node` 的 ROS 转换和诊断；
5. 标定并配置 `station_model→station_assembly`；
6. 实现 tf，输出 `arm_base → station_assembly`；
7. 关闭仿真真值 pose，验证视觉结果进入 planning。

在 3D 点、手眼外参和模型系到装配系变换都经过实测前，不应把该链路用于真实机械臂。

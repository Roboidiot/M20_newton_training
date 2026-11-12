"""仅依赖 PyTorch 的通用轮模型部署模块；可单独复制到 IsaacLab 项目。

与 best.pt 一起使用，不导入本项目训练/数据/NumPy 代码。
所有输入/输出张量留在模型设备，模型只做冻结推理。
输入为原始单位、未经对称变换的世界系数据；内部自动变换、标准化与恢复输出。
具体数据含义、沙面来源和物理步调用方式见 DEPLOY_ISAACLAB.md。
"""

import math

import torch
from torch import nn

WHEELS = ("FL", "FR", "HL", "HR")
FEATURE_NAMES = [
    "sinkage_reference_m",
    "sinkage_local_m",
    "surface_valid",
    "center_vx_w_m_s",
    "center_vy_w_m_s",
    "center_vz_w_m_s",
    "omega_x_w_rad_s",
    "omega_y_w_rad_s",
    "omega_z_w_rad_s",
    "axis_x_w",
    "axis_y_w",
    "axis_z_w",
    "wheel_joint_velocity_rad_s",
    "previous_center_vx_w_m_s",
    "previous_center_vy_w_m_s",
    "previous_center_vz_w_m_s",
    "previous_omega_x_w_rad_s",
    "previous_omega_y_w_rad_s",
    "previous_omega_z_w_rad_s",
]
TARGET_NAMES = ["Fx_N", "Fy_N", "Fz_N", "Tx_com_Nm", "Ty_com_Nm", "Tz_com_Nm"]
SYMMETRY = {
    "FL": {"polar_diagonal": [1, 1, 1], "axis_gauge": 1},
    "FR": {"polar_diagonal": [1, -1, 1], "axis_gauge": -1},
    "HL": {"polar_diagonal": [-1, 1, 1], "axis_gauge": 1},
    "HR": {"polar_diagonal": [-1, -1, 1], "axis_gauge": -1},
}


class WheelModelRuntime(nn.Module):
    """一个共享权重预测四轮：[...,4,19] → [...,4,6]。

    常见调用 model(x) 中 x 为 [num_envs,4,19]，轮序 FL/FR/HL/HR。
    predict_wheel(x, 'FR') 支持单轮 [...,19] → [...,6]。
    返回世界系沙力 N 与绕对应轮 MPM 代理输入质心的力矩 N·m。
    """

    def __init__(self, checkpoint_path, device="cpu"):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("model_type") != "shared_wheel_symmetry_v1"
            or checkpoint.get("feature_names") != FEATURE_NAMES
            or checkpoint.get("target_names") != TARGET_NAMES
            or checkpoint.get("symmetry") != SYMMETRY
            or checkpoint.get("wheel_order") != list(WHEELS)
        ):
            raise ValueError(
                "Checkpoint is not a compatible shared 19-input wheel model"
            )
        history = checkpoint.get("history", {})
        self.physics_dt = float(history.get("dt_s", -1))
        if (
            history.get("steps") != 1
            or not math.isfinite(self.physics_dt)
            or self.physics_dt <= 0
        ):
            raise ValueError("Expected one-step history and a positive physics dt")
        self.best_epoch = checkpoint.get("epoch")

        sizes = [19, *checkpoint["hidden"], 6]
        layers = []
        for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
        self.network.load_state_dict(checkpoint["model_state"])
        for prefix, key in (("x", "x_scaler"), ("y", "y_scaler")):
            for item in ("mean", "scale"):
                values = torch.tensor(checkpoint[key][item], dtype=torch.float32)
                expected = 19 if prefix == "x" else 6
                if values.shape != (expected,) or not torch.isfinite(values).all():
                    raise ValueError(f"Invalid {key}.{item}")
                if item == "scale" and (values <= 0).any():
                    raise ValueError("Scaler values must be positive")
                self.register_buffer(f"{prefix}_{item}", values)

        input_sign = torch.ones(4, 19)
        output_sign = torch.ones(4, 6)
        for i, wheel in enumerate(WHEELS):
            rule = SYMMETRY[wheel]
            polar = torch.tensor(rule["polar_diagonal"], dtype=torch.float32)
            determinant = int(polar.prod().item())
            axial = determinant * polar
            input_sign[i, 3:6] = input_sign[i, 13:16] = polar
            input_sign[i, 6:9] = input_sign[i, 16:19] = axial
            input_sign[i, 9:12] = rule["axis_gauge"] * polar
            input_sign[i, 12] = determinant * rule["axis_gauge"]
            output_sign[i, :3], output_sign[i, 3:] = polar, axial
        self.register_buffer("input_sign", input_sign)
        self.register_buffer("output_sign", output_sign)
        self.to(device=device)
        self.requires_grad_(False)
        self.eval()

    def check_physics_dt(self, dt):
        """初始化时核对调用周期；历史必须是一个训练物理步，不是一个策略步。"""
        if not math.isclose(float(dt), self.physics_dt, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Model history dt={self.physics_dt}, simulation dt={dt}")

    def _raw_input(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[-1] != 19:
            raise ValueError("Expected a Torch tensor with last dimension 19")
        if x.device != self.x_mean.device:
            raise ValueError(
                f"Input device {x.device} differs from model device {self.x_mean.device}"
            )
        return x.to(dtype=torch.float32)

    def _canonical_prediction(self, aligned):
        # float32 与旧 NumPy 实现的 float64 中间缩放可能有微小舍入差异。
        scaled = (aligned - self.x_mean) / self.x_scale
        return self.network(scaled) * self.y_scale + self.y_mean

    @torch.no_grad()
    def forward(self, x):
        x = self._raw_input(x)
        if x.ndim < 2 or x.shape[-2] != 4:
            raise ValueError(
                "Expected four-wheel features [...,4,19] in FL/FR/HL/HR order"
            )
        return self._canonical_prediction(x * self.input_sign) * self.output_sign

    @torch.no_grad()
    def predict_wheel(self, x, wheel):
        if wheel not in WHEELS:
            raise ValueError(f"Unknown wheel {wheel!r}")
        i = WHEELS.index(wheel)
        x = self._raw_input(x)
        return self._canonical_prediction(x * self.input_sign[i]) * self.output_sign[i]


class WheelMotionHistory:
    """每个并行环境独立维护四轮上一物理步的六维速度。

    append(current13) 返回完整 x19 及 ready[num_envs]。
    ready=False 表示本次缺少历史；调用方应跳过该环境此步代理力或采用自己的过渡方案。
    reset(env_ids) 只清除被重置的环境，不污染仍在运行的环境。
    """

    def __init__(self, num_envs, device="cpu"):
        self.previous = torch.zeros(num_envs, 4, 6, dtype=torch.float32, device=device)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @torch.no_grad()
    def reset(self, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        self.previous[indices] = 0
        self.ready[indices] = False

    @torch.no_grad()
    def append(self, current13):
        if (
            current13.shape != (*self.previous.shape[:2], 13)
            or current13.device != self.previous.device
        ):
            raise ValueError(
                "Expected current features [num_envs,4,13] on the history device"
            )
        current13 = current13.to(dtype=torch.float32)
        valid = self.ready.clone()
        x = torch.cat((current13, self.previous), dim=-1)
        # cat 已复制历史，再更新缓存；同一物理步只调用一次。
        self.previous.copy_(current13[..., 3:9])
        self.ready.fill_(True)
        return x, valid


def build_current_features(
    reference_height_w,
    local_height_w,
    local_valid,
    bottom_z_w,
    center_velocity_w,
    angular_velocity_w,
    axis_w,
    joint_velocity,
):
    """按训练列顺序构造 [num_envs,4,13] 的当前特征，所有张量需在同一设备。

    reference_height_w: [num_envs,1]（每环境初始沙面）或 [num_envs,4]。
    local_height_w/local_valid/bottom_z_w/joint_velocity: [num_envs,4]。
    三种向量为 [num_envs,4,3]。局部沙面无效时 local_height 可为 NaN，输出局部深度填 0。
    下陷量不截断，允许负值。
    """
    if bottom_z_w.ndim != 2 or bottom_z_w.shape[-1] != 4:
        raise ValueError("Expected bottom_z_w [num_envs,4]")
    if reference_height_w.shape not in ((bottom_z_w.shape[0], 1), bottom_z_w.shape):
        raise ValueError("Reference heights must be [num_envs,1] or [num_envs,4]")
    valid = local_valid.to(dtype=torch.bool)
    reference_depth = reference_height_w - bottom_z_w
    local_depth = torch.where(
        valid, local_height_w - bottom_z_w, torch.zeros_like(bottom_z_w)
    )
    return torch.cat(
        (
            reference_depth[..., None],
            local_depth[..., None],
            valid.to(dtype=torch.float32)[..., None],
            center_velocity_w,
            angular_velocity_w,
            axis_w,
            joint_velocity[..., None],
        ),
        dim=-1,
    ).float()


def shift_torque_reference(force_w, torque_at_source_w, source_com_w, target_com_w):
    """将世界系力矩从模型代理输入质心 P 平移到物理引擎施力质心 C。"""
    return torque_at_source_w + torch.linalg.cross(
        source_com_w - target_com_w, force_w, dim=-1
    )

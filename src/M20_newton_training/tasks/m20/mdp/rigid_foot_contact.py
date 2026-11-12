"""Expose measured rigid contact forces to the shared 50 Hz foot-gait rewards.

No external wrench, fitted sand model or policy action dimension is introduced.
"""

import torch

from isaaclab.utils.math import quat_apply

from .proxy_contact_rewards import ProxyFootContactState

FOOT_NAMES = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")


class RigidFootContact:
    """Read world-Z support forces once per control step, shared by all gait terms."""

    foot_names = FOOT_NAMES

    def __init__(self, env, sensor_name):
        self.env = env
        self.sensor = env.scene.sensors[sensor_name]
        self.robot = env.scene["robot"]
        self.sensor_ids, names = self.sensor.find_bodies(list(FOOT_NAMES), preserve_order=True)
        if list(names) != list(FOOT_NAMES):
            raise ValueError(f"Expected ordered rigid sensor feet {FOOT_NAMES}; got {names}")
        self.body_ids, names = self.robot.find_bodies(list(FOOT_NAMES), preserve_order=True)
        if list(names) != list(FOOT_NAMES):
            raise ValueError(f"Expected ordered robot feet {FOOT_NAMES}; got {names}")
        self.contact_state = ProxyFootContactState(env.num_envs, env.device)
        self.application_points_w = torch.zeros((env.num_envs, 4, 3), device=env.device)
        self.local_axis = torch.tensor([0.0, -1.0, 0.0], device=env.device).expand(env.num_envs, 4, 3)
        self.last_step = None

    @torch.no_grad()
    def update(self):
        step = self.env.common_step_counter
        if step == self.last_step:
            return self
        # Sensor data already includes rigid normal/friction reactions; no proxy calculation.
        forces = self.sensor.data.net_forces_w
        forces = forces if isinstance(forces, torch.Tensor) else forces.torch
        self.contact_state.begin_step()
        self.contact_state.update(forces[:, self.sensor_ids], self.env.step_dt, 5.0)
        data = self.robot.data
        self.application_points_w.copy_(data.body_link_pos_w.torch[:, self.body_ids])
        axis = quat_apply(data.body_link_quat_w.torch[:, self.body_ids], self.local_axis)
        # M20 USD wheel capsules: radius 9 cm, cylinder half-length 2.7 cm.
        self.application_points_w[..., 2] -= 0.09 + 0.027 * axis[..., 2].abs()
        self.last_step = step
        return self

    def reset(self, env_ids=None):
        self.contact_state.reset(env_ids)


def rigid_foot_contacts(env, sensor_name="contact_forces"):
    providers = getattr(env, "_rigid_foot_contact_providers", None)
    if providers is None:
        providers = env._rigid_foot_contact_providers = {}
    if sensor_name not in providers:
        providers[sensor_name] = RigidFootContact(env, sensor_name)
    return providers[sensor_name]

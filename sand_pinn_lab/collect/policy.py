"""Slow random excitation and optional deterministic RSL-RL actor inference."""

import numpy as np


class SlowRandomPolicy:
    def __init__(self, seed, action_dim=16, limit=1.5, hold=(0.5, 1.0), transition=0.2):
        self.rng = np.random.default_rng(seed)
        self.limit, self.hold, self.transition = limit, hold, transition
        self.current = np.zeros(action_dim, dtype=np.float32)
        self.target = self.current.copy()
        self.start = self.current.copy()
        self.changed_at = 0.0
        self.next_change = 0.0
        self.segment = -1

    def __call__(self, time_s):
        if time_s + 1e-9 >= self.next_change:
            self.start = self.current.copy()
            self.target = self.rng.uniform(-self.limit, self.limit, len(self.current)).astype(np.float32)
            self.changed_at = time_s
            self.next_change = time_s + self.rng.uniform(*self.hold)
            self.segment += 1
        alpha = 1.0 if self.transition == 0 else np.clip((time_s - self.changed_at) / self.transition, 0, 1)
        alpha = alpha * alpha * (3 - 2 * alpha)
        self.current = (self.start + alpha * (self.target - self.start)).astype(np.float32)
        return self.current.copy()


class CheckpointPolicy:
    """Load only the saved actor, including its observation normalizer; never run PPO.

    Random-only collection does not import Torch/RSL-RL through this module.
    The legacy random_target field stores the executed policy action and
    random_segment is -1; metadata identifies these compatibility fields.
    """

    def __init__(self, checkpoint, agent_config, device="cpu"):
        import torch
        import yaml
        from rsl_rl.models import MLPModel
        from tensordict import TensorDict

        class ConfigLoader(yaml.SafeLoader):
            pass

        ConfigLoader.add_constructor(
            "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
        )
        with open(agent_config) as stream:
            config = yaml.load(stream, Loader=ConfigLoader)
        actor_cfg = dict(config["actor"])
        if actor_cfg.pop("class_name") != "MLPModel":
            raise ValueError("Collection supports the M20 MLPModel actor only")
        self.obs_groups = config["obs_groups"]
        if self.obs_groups.get("actor") != ["policy"]:
            raise ValueError("Expected the M20 actor observation group ['policy']")
        self.device = device
        sample = TensorDict({"policy": torch.zeros(1, 57, device=device)}, batch_size=[1])
        self.actor = MLPModel(sample, self.obs_groups, "actor", 16, **actor_cfg).to(device)
        loaded = torch.load(checkpoint, map_location=device, weights_only=True)
        self.actor.load_state_dict(loaded["actor_state_dict"], strict=True)
        self.actor.eval()
        self.actor.requires_grad_(False)
        self.clip_actions = config.get("clip_actions")
        if self.clip_actions is not None and (not np.isfinite(self.clip_actions) or self.clip_actions <= 0):
            raise ValueError("clip_actions must be positive and finite or null")
        self.target = np.zeros(16, dtype=np.float32)
        self.segment = -1

    def __call__(self, observations):
        import torch
        from tensordict import TensorDict

        obs = observations["policy"]
        if tuple(obs.shape) != (1, 57) or not torch.isfinite(obs).all():
            raise ValueError("Policy requires finite observations with shape [1, 57]")
        with torch.inference_mode():
            action = self.actor(TensorDict({"policy": obs.to(self.device)}, batch_size=[1]), stochastic_output=False)
            if self.clip_actions is not None:
                action = action.clamp(-self.clip_actions, self.clip_actions)
            if not torch.isfinite(action).all():
                raise FloatingPointError("Checkpoint actor returned nonfinite actions")
        self.target = action[0].cpu().numpy().astype(np.float32, copy=True)
        return self.target.copy()


def set_command(raw, command):
    """Publish this step's fixed command before constructing actor observations."""
    import torch

    term = raw.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = torch.as_tensor(command, device=term.vel_command_b.device)
    term.is_standing_env.fill_(not any(command))
    term.is_heading_env.zero_()

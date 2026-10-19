# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class DeeproboticsM20RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "deeprobotics_m20_rough"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 100
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        """Hook for robot-specific runner settings; configclass handles field copying."""


@configclass
class DeeproboticsM20FlatPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "deeprobotics_m20_flat"


@configclass
class DeeproboticsM20SandPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "deeprobotics_m20_sand"


@configclass
class DeeproboticsM20StandPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "deeprobotics_m20_stand"
        self.max_iterations = 20000

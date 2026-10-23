"""Keep contact-dependent Torch actions outside Newton's folded policy-step loop."""

from isaaclab.envs import ManagerBasedRLEnv


class FootSandProxyEnv(ManagerBasedRLEnv):
    """Recompute the external sand force at every physics step, including graph replay.

    Newton may fold all actuator substeps into one policy-step graph. The sand
    action depends on current foot state, so the environment must own decimation
    and call ActionManager.apply_action() before each individual physics step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim.physics_manager.set_decimation(1)
        self._physics_handles_decimation = False

"""Critic warmup using the installed RSL-RL PPO implementation unchanged."""

from __future__ import annotations

from rsl_rl.algorithms import PPO


class ActorWarmupPPO(PPO):
    """Freeze the complete actor for the first N successful PPO updates.

    Rollout collection remains stochastic. The inherited PPO loss still updates
    the critic, while actor gradients (including the distribution's log-std) are
    disabled. The optimizer retains its parameter groups and critic moments when
    the actor is unfrozen. Actor observation statistics and adaptive KL-based
    learning-rate changes are paused during warmup as well.
    """

    def __init__(self, *args, actor_warmup_iterations: int = 0, **kwargs):
        if type(actor_warmup_iterations) is not int or actor_warmup_iterations < 0:
            raise ValueError("actor_warmup_iterations must be a nonnegative integer")
        super().__init__(*args, **kwargs)
        self.actor_warmup_iterations = actor_warmup_iterations
        self.completed_updates = 0
        self._actor_grad_flags = tuple(p.requires_grad for p in self._raw_actor.parameters())
        if {id(p) for p in self._raw_actor.parameters()} & {id(p) for p in self._raw_critic.parameters()}:
            raise ValueError("Actor warmup requires separate actor and critic parameters")
        self._apply_actor_freeze()

    @property
    def actor_frozen(self) -> bool:
        return self.completed_updates < self.actor_warmup_iterations

    def _apply_actor_freeze(self):
        for parameter, trainable in zip(self._raw_actor.parameters(), self._actor_grad_flags, strict=True):
            parameter.requires_grad_(trainable and not self.actor_frozen)
            # None, rather than a zero gradient, prevents Adam's restored momentum
            # and any weight decay from moving a frozen actor.
            if self.actor_frozen:
                parameter.grad = None

    def update(self) -> dict[str, float]:
        frozen = self.actor_frozen
        schedule = self.schedule
        normalization = self._raw_actor.obs_normalization
        if frozen:
            self.schedule = "fixed"
            # MLPModel.get_latent still uses the existing normalizer; this flag
            # controls only update_normalization(), which runs at the end of PPO.
            self._raw_actor.obs_normalization = False
        try:
            losses = super().update()
        finally:
            self.schedule = schedule
            self._raw_actor.obs_normalization = normalization
        self.completed_updates += 1
        self._apply_actor_freeze()
        losses["actor_frozen"] = float(frozen)
        if frozen and not self.actor_frozen:
            print(f"[ActorWarmupPPO] Critic warmup completed after {self.completed_updates} updates; actor unfrozen.")
        return losses

    def save(self) -> dict:
        saved = super().save()
        saved["actor_warmup_state"] = {"completed_updates": self.completed_updates}
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resume_iteration = super().load(loaded_dict, load_cfg, strict)
        if resume_iteration:
            state = loaded_dict.get("actor_warmup_state")
            # Native checkpoints label the first completed PPO update as iter=0.
            self.completed_updates = (
                int(state["completed_updates"]) if state is not None else max(0, int(loaded_dict.get("iter", -1)) + 1)
            )
        else:
            # Loading weights for a new run starts a new warmup, unlike resuming.
            self.completed_updates = 0
        self._apply_actor_freeze()
        return resume_iteration

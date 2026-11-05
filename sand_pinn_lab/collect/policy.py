"""Seeded, slow random joint-target excitation; independent of RL libraries."""

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

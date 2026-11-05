"""Independent M20 sand data collection and physics configuration interface."""

from .physics import PhysicsConfig, apply_physics_config, load_physics_config

__all__ = ["PhysicsConfig", "apply_physics_config", "load_physics_config"]

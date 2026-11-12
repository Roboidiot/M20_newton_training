"""Compatibility imports for the preserved analytical calibration/shadow tools.

Training selects its backend through FootSandProxyActionCfg.model_kind.
"""

from .analytical.foot_model import DEFAULT_MODEL_PATH, FootSandModel, power_response

__all__ = ["DEFAULT_MODEL_PATH", "FootSandModel", "power_response"]

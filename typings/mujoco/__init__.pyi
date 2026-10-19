"""Public MuJoCo bindings used by the CPU replay tool.

MuJoCo 3.12's wheel exports these symbols from binary modules without stubs.
This file supplies their public signatures for Pylance; it has no runtime effect.
"""

from enum import IntEnum
from typing import Self

import numpy as np
from numpy.typing import NDArray

class mjtObj(IntEnum):
    mjOBJ_BODY = ...
    mjOBJ_JOINT = ...
    mjOBJ_ACTUATOR = ...

class mjtIntegrator(IntEnum):
    mjINT_IMPLICITFAST = ...

class mjtCone(IntEnum):
    mjCONE_PYRAMIDAL = ...

class mjtCamera(IntEnum):
    mjCAMERA_TRACKING = ...

class MjOption:
    timestep: float
    integrator: int
    cone: int

class MjModel:
    opt: MjOption
    ngeom: int
    jnt_qposadr: NDArray[np.int32]
    jnt_dofadr: NDArray[np.int32]
    actuator_trnid: NDArray[np.int32]
    actuator_gear: NDArray[np.float64]
    dof_damping: NDArray[np.float64]
    dof_frictionloss: NDArray[np.float64]
    dof_armature: NDArray[np.float64]
    geom_contype: NDArray[np.int32]
    geom_conaffinity: NDArray[np.int32]
    geom_bodyid: NDArray[np.int32]
    @classmethod
    def from_xml_path(cls, filename: str, assets: dict[str, bytes] | None = ...) -> Self: ...

class MjData:
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    ctrl: NDArray[np.float64]
    xmat: NDArray[np.float64]
    xpos: NDArray[np.float64]
    time: float
    def __init__(self, model: MjModel) -> None: ...

def mj_name2id(model: MjModel, type: int, name: str) -> int: ...
def mj_resetData(model: MjModel, data: MjData) -> None: ...
def mj_forward(model: MjModel, data: MjData) -> None: ...
def mj_step(model: MjModel, data: MjData, nstep: int = ...) -> None: ...
def mj_objectVelocity(
    model: MjModel, data: MjData, objtype: int, objid: int, res: NDArray[np.float64], flg_local: int
) -> None: ...

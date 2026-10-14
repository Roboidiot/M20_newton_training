# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset configurations for M20_newton_training."""

import os

# Path to the project's top-level ``deep_robotics_model`` data directory.
# The robot models live outside the Python package (binary USD/STL assets).
ASSETS_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "deep_robotics_model"))

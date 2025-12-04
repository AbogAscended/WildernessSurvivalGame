"""Environment namespace package.

Exports the Gymnasium-compatible Wilderness Survival System environment.

Example
-------
>>> from src.envs import WildernessSurvivalEnv
>>> env = WildernessSurvivalEnv(width=20, height=10, difficulty="medium")
"""

from .wss_env import WildernessSurvivalEnv

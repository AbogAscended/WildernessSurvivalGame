"""Centralized difficulty specification for WSS.

This module defines difficulty labels and provides parameterized probabilities
and ranges for several subsystems:

- Terrain weights (used by :func:`src.Terrain.random_terrain`)
- Item spawning (per-tile probabilities), repeating chances, and amount ranges
- Trader behavior profiles

Difficulties supported: ``"easy"``, ``"medium"``, ``"hard"``, ``"extreme"``.
For backward compatibility, the alias ``"normal"`` maps to ``"medium"``.

Also provided is a numeric hardness scalar ``h ∈ [0,1]`` for RL curricula.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple


DIFFICULTY_ORDER = ("easy", "medium", "hard", "extreme")


def canonicalize(name: str) -> str:
    """Normalize a difficulty label to one of the canonical names.

    :param str name: Input label (e.g., ``"normal"`` or any known label).
    :return: Canonical difficulty in {``easy``, ``medium``, ``hard``,
        ``extreme``}. Unknown values map to ``"medium"``; ``"normal"`` maps
        to ``"medium"``.
    :rtype: str
    """
    if not isinstance(name, str):
        return "medium"
    n = name.strip().lower()
    if n == "normal":
        return "medium"
    return n if n in DIFFICULTY_ORDER else "medium"


# Simple scalar hardness for RL-friendly scaling (0..1)
_HARDNESS = {
    "easy": 0.0,
    "medium": 1.0/3.0,
    "hard": 2.0/3.0,
    "extreme": 1.0,
}


def get_hardness(name: str) -> float:
    """Get a scalar hardness value in ``[0, 1]`` for a difficulty label.

    :param str name: Difficulty label.
    :return: Hardness scalar (``0.0`` for easy, ``1.0`` for extreme, with
        evenly spaced values in between).
    :rtype: float
    """
    return float(_HARDNESS.get(canonicalize(name), _HARDNESS["medium"]))


# Terrain weights are ordered like Terrain.TERRAIN_TYPES
_TERRAIN_WEIGHTS: Dict[str, List[int]] = {
    # plains, forest, swamp, mountain, desert
    "easy":   [6, 3, 1, 0, 0],
    "medium": [4, 3, 1, 1, 1],
    "hard":   [2, 2, 2, 2, 2],
    "extreme":[1, 1, 3, 3, 3],
}


def get_terrain_weights(difficulty: str) -> List[int]:
    """Return per-type weights for terrain sampling.

    The order matches :data:`src.Terrain.TERRAIN_TYPES` (plains, forest,
    swamp, mountain, desert).

    :param str difficulty: Difficulty label.
    :return: List of non-negative weights.
    :rtype: list[int]
    """
    return _TERRAIN_WEIGHTS.get(canonicalize(difficulty), _TERRAIN_WEIGHTS["medium"])


# Item spawn configuration per difficulty (probs, ranges)
_ITEM_CONFIG: Dict[str, Dict[str, object]] = {
    "easy": {
        "trader_prob": 0.03,
        "water_prob": 0.12,
        "food_prob": 0.10,
        "gold_prob": 0.05,
        "water_repeat_prob": 0.7,
        "food_repeat_prob": 0.5,
        "water_amount": (3, 6),
        "food_amount": (3, 6),
        "gold_amount": (4, 8),
    },
    "medium": {
        "trader_prob": 0.05,
        "water_prob": 0.08,
        "food_prob": 0.06,
        "gold_prob": 0.04,
        "water_repeat_prob": 0.5,
        "food_repeat_prob": 0.35,
        "water_amount": (2, 5),
        "food_amount": (2, 5),
        "gold_amount": (3, 6),
    },
    "hard": {
        "trader_prob": 0.06,
        "water_prob": 0.05,
        "food_prob": 0.04,
        "gold_prob": 0.03,
        "water_repeat_prob": 0.35,
        "food_repeat_prob": 0.25,
        "water_amount": (2, 4),
        "food_amount": (2, 4),
        "gold_amount": (2, 5),
    },
    "extreme": {
        "trader_prob": 0.08,
        "water_prob": 0.03,
        "food_prob": 0.02,
        "gold_prob": 0.02,
        "water_repeat_prob": 0.2,
        "food_repeat_prob": 0.1,
        "water_amount": (1, 3),
        "food_amount": (1, 3),
        "gold_amount": (1, 4),
    },
}


def get_item_config(difficulty: str) -> Dict[str, object]:
    """Get item spawning config for a difficulty.

    The dictionary contains keys like ``trader_prob``, ``water_prob``,
    ``food_prob``, ``gold_prob``, repeating chances, and amount ranges.

    :param str difficulty: Difficulty label.
    :return: Copy of the item configuration dict for mutation safety.
    :rtype: dict
    """
    return _ITEM_CONFIG.get(canonicalize(difficulty), _ITEM_CONFIG["medium"]).copy()


# Trader behavior profile per difficulty.
# We keep the existing "deviation" logic (0.5, 1.0, 1.5) but vary the mix.
_TRADER_PROFILE: Dict[str, Dict[str, object]] = {
    "easy": {
        "deviation_weights": {1.5: 0.6, 1.0: 0.35, 0.5: 0.05},
        "max_counters": {1.5: 4, 1.0: 3, 0.5: 2},
        "total_range": (3, 6),
    },
    "medium": {
        "deviation_weights": {1.5: 0.5, 1.0: 0.4, 0.5: 0.1},
        "max_counters": {1.5: 4, 1.0: 3, 0.5: 2},
        "total_range": (2, 5),
    },
    "hard": {
        "deviation_weights": {1.5: 0.3, 1.0: 0.5, 0.5: 0.2},
        "max_counters": {1.5: 3, 1.0: 2, 0.5: 1},
        "total_range": (2, 4),
    },
    "extreme": {
        "deviation_weights": {1.5: 0.15, 1.0: 0.35, 0.5: 0.5},
        "max_counters": {1.5: 2, 1.0: 1, 0.5: 1},
        "total_range": (1, 3),
    },
}


def get_trader_profile(difficulty: str) -> Dict[str, object]:
    """Get trader behavior profile for a difficulty.

    Keys include ``deviation_weights`` (probabilities over deviation levels),
    ``max_counters`` (per deviation), and ``total_range`` for proposal sums.

    :param str difficulty: Difficulty label.
    :return: Copy of the profile dictionary.
    :rtype: dict
    """
    return _TRADER_PROFILE.get(canonicalize(difficulty), _TRADER_PROFILE["medium"]).copy()


def sample_deviation(rng: random.Random,
                     profile: Dict[str, object]
                     ) -> float:
    """Sample a deviation level from a trader profile.

    :param random.Random rng: RNG used for sampling.
    :param dict profile: Trader profile dict with a ``deviation_weights`` map.
    :return: Selected deviation level.
    :rtype: float
    """
    weights = profile["deviation_weights"]
    choices = list(weights.keys())
    probs = [weights[k] for k in choices]
    # cumulative choice
    r = rng.random()
    cum = 0.0
    for c, p in zip(choices, probs):
        cum += p
        if r <= cum:
            return float(c)
    return float(choices[-1])


MAP_SIZES: Dict[str, Tuple[int, int]] = {
    "easy": (30, 50),
    "medium": (50, 70),
    "hard": (70, 90),
    "extreme": (90, 110),
}

VISION_RADII: Dict[str, int] = {
    "easy": 4,
    "medium": 3,
    "hard": 2,
    "extreme": 1,
}

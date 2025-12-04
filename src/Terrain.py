"""Terrain definitions and difficulty-weighted random selection.

This module defines the base ``Terrain`` class used by tiles and utilities to
sample random terrain according to the current difficulty. Difficulties and
their weights are provided by ``src.Difficulty`` when available.

Public API
----------
- :class:`Terrain` — immutable-like container for terrain costs
- :func:`random_terrain` — sample a terrain using difficulty-weighted choice
"""

import random
# Try package-relative import first; fall back to absolute when run as a script
try:  # pragma: no cover - import resolution shim
    from . import Difficulty  # type: ignore
except Exception:  # pragma: no cover
    try:
        import Difficulty  # type: ignore
    except Exception:
        Difficulty = None  # type: ignore

class Terrain:
    """Terrain object used by the map/tiles.

    Parameters
    ----------
    name : str
        Human-readable terrain name (e.g., ``"plains"``, ``"forest"``).
    move_cost : int
        Strength/movement points required to enter the tile.
    water_cost : int
        Water units consumed on entering the tile.
    food_cost : int
        Food units consumed on entering the tile.

    Notes
    -----
    Instances are lightweight data holders. All costs are applied when the
    player enters the tile.
    """

    def __init__(self,
                 name,
                 move_cost,
                 water_cost,
                 food_cost
                 ):
        self.name = name
        self.move_cost = move_cost
        self.water_cost = water_cost
        self.food_cost = food_cost

    def costs(self):
        """Get the cost tuple for entering this terrain.

        Returns
        -------
        tuple[int, int, int]
            ``(move_cost, water_cost, food_cost)``.
        """
        return self.move_cost, self.water_cost, self.food_cost

    def can_enter(self,
                  strength,
                  water,
                  food):
        """Check if a player can enter this terrain with current resources.

        Parameters
        ----------
        strength : int
            Player's current strength/movement points.
        water : int
            Player's current water units.
        food : int
            Player's current food units.

        Returns
        -------
        bool
            ``True`` if the player has enough of each resource to pay costs on
            entry; otherwise ``False``.
        """
        m, w, f = self.costs()
        return strength >= m and water >= w and food >= f

    def __repr__(self):
        # mainly for debugging
        return f"Terrain({self.name}, move={self.move_cost}, water={self.water_cost}, food={self.food_cost})"

# base terrain types
PLAINS   = Terrain("plains",   move_cost=1, water_cost=1, food_cost=1)
FOREST   = Terrain("forest",   move_cost=2, water_cost=1, food_cost=2)
SWAMP    = Terrain("swamp",    move_cost=3, water_cost=2, food_cost=2)
MOUNTAIN = Terrain("mountain", move_cost=4, water_cost=3, food_cost=3)
DESERT   = Terrain("desert",   move_cost=2, water_cost=4, food_cost=3)

# list so we can randomly pick one
TERRAIN_TYPES = [PLAINS, FOREST, SWAMP, MOUNTAIN, DESERT]

def random_terrain(difficulty="normal"):
    """Pick a random :class:`Terrain` based on difficulty.

    Parameters
    ----------
    difficulty : str, optional
        Difficulty label in {``"easy"``, ``"medium"``, ``"hard"``,
        ``"extreme"``}. The legacy alias ``"normal"`` is accepted and maps to
        ``"medium"``. Unknown values fall back to ``"medium"``.

    Returns
    -------
    Terrain
        A newly instantiated terrain object (not one of the shared constants) so
        tiles do not share identity.
    """
    if Difficulty is not None:  # type: ignore
        weights = Difficulty.get_terrain_weights(difficulty)  # type: ignore
    else:
        # Fallback weights if Difficulty module is unavailable
        fallback = {
            "easy":    [6, 3, 1, 0, 0],
            "medium":  [4, 3, 1, 1, 1],
            "hard":    [2, 2, 2, 2, 2],
            "extreme": [1, 1, 3, 3, 3],
        }
        key = difficulty if difficulty in fallback else ("medium" if difficulty != "easy" else "easy")
        weights = fallback[key]
    base = random.choices(TERRAIN_TYPES, weights=weights, k=1)[0]

    # make a new Terrain so tiles don't all share the same object
    return Terrain(base.name, base.move_cost, base.water_cost, base.food_cost)

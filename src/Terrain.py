# stores the different terrain types & their costs
# each terrain has move cost, water cost & food cost

import random

class Terrain:
    """
    terrain object used by the map/tiles

    name - string like "plains" or "forest"
    move_cost - strength / movement cost to enter
    water_cost - water cost to enter
    food_cost - food cost to enter
    """

    def __init__(self, name, move_cost, water_cost, food_cost):
        self.name = name
        self.move_cost = move_cost
        self.water_cost = water_cost
        self.food_cost = food_cost

    def costs(self):
        """return (move_cost, water_cost, food_cost)"""
        return self.move_cost, self.water_cost, self.food_cost

    def can_enter(self, strength, water, food):
        """
        check if a player can enter this terrain
        with the given resources
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

# difficulty weights (same order as TERRAIN_TYPES)
# bigger numbers = more likely
_DIFFICULTY_WEIGHTS = {
    "easy":    [6, 3, 1, 0, 0],  # mostly plains/forest
    "normal":  [4, 3, 1, 1, 1],  # mix of everything
    "hard":    [2, 2, 2, 2, 2],  # pretty even
    "extreme": [1, 1, 3, 3, 3],  # lots of bad tiles
}

def random_terrain(difficulty="normal"):
    """
    pick a random Terrain based on difficulty

    difficulty: "easy", "normal", "hard", "extreme"
    unknown difficulty falls back to "normal"
    """
    weights = _DIFFICULTY_WEIGHTS.get(difficulty, _DIFFICULTY_WEIGHTS["normal"])
    base = random.choices(TERRAIN_TYPES, weights=weights, k=1)[0]

    # make a new Terrain so tiles don't all share the same object
    return Terrain(base.name, base.move_cost, base.water_cost, base.food_cost)

# one tile on the map
# has a terrain & maybe items/trader on it

import Terrain
import Items
import Trader


class Tile:
    """
    tile on the map grid

    terrain - Terrain.Terrain obj
    items - list of Items or a Trader on the tile
    """

    def __init__(self, terrain=None, difficulty="normal"):
        # if terrain is not given, just pick a random one
        if terrain is None:
            terrain = Terrain.random_terrain(difficulty)

        self.terrain = terrain
        self.items = []  # items that exist on this tile

    #terrain

    def get_costs(self):
        """return (move_cost, water_cost, food_cost)"""
        return self.terrain.costs()

    def is_passable(self, strength, water, food):
        """
        check if player has enough resources to walk onto this tile
        doesn't actually move them, just checks
        """
        return self.terrain.can_enter(strength, water, food)

    #item / trader

    def add_item(self, item):
        """put an item or trader on this tile"""
        if item is not None:
            self.items.append(item)

    def has_trader(self):
        """true if any trader is on this tile"""
        return any(isinstance(obj, Trader.Trader) for obj in self.items)

    def collect_items(self, player):
        """
        give non trader items to the player

        repeating items stay forever (streams)
        one time items disappear after being used
        """
        keep = []

        for item in self.items:
            # traders aren't collected here
            if isinstance(item, Trader.Trader):
                keep.append(item)
                continue

            amt = item.getAmount()
            if amt > 0:
                player.collect(item)

            # repeating items stay on tile
            if item.isRepeating and amt > 0:
                keep.append(item)

        self.items = keep

    def __repr__(self):
        # for debugging/printing the map
        return f"Tile({self.terrain.name}, items={len(self.items)})"

"""Tile module.

Represents a single cell on the map grid. A tile always has a
``Terrain`` instance and may contain one or more ``Items`` or a
``Trader``.
"""

# Prefer package-relative imports; fall back to absolute when modules are run directly
try:  # pragma: no cover - import resolution shim
    from . import Terrain, Items, Trader  # type: ignore
except Exception:  # pragma: no cover
    import Terrain  # type: ignore
    import Items  # type: ignore
    import Trader  # type: ignore


class Tile:
    """A single map cell holding a terrain and optional items/trader.

    :param Terrain.Terrain terrain: Terrain for this tile. If ``None``, a
        random terrain is selected using the current difficulty.
    :param str difficulty: Difficulty label passed to :func:`Terrain.random_terrain`
        when ``terrain`` is ``None``.

    :ivar Terrain.Terrain terrain: Terrain associated with the tile.
    :ivar list items: List of :class:`Items.Items` (including :class:`Trader.Trader`).
    """

    def __init__(self, terrain=None, difficulty="normal"):
        """Construct a tile.

        :param terrain: Optional pre-created :class:`Terrain.Terrain`.
        :param str difficulty: Difficulty string used if ``terrain`` is ``None``.
        """
        # if terrain is not given, just pick a random one
        if terrain is None:
            terrain = Terrain.random_terrain(difficulty)

        self.terrain = terrain
        self.difficulty = difficulty
        self.items = []  # items that exist on this tile

    def get_costs(self):
        """Get terrain entry costs for this tile.

        :return: Tuple ``(move_cost, water_cost, food_cost)``.
        :rtype: tuple[int, int, int]
        """
        return self.terrain.costs()

    def is_passable(self, strength, water, food):
        """Check if the player can afford to enter this tile.

        This does not change player state; it only checks costs.

        :param int strength: Player's current strength.
        :param int water: Player's current water.
        :param int food: Player's current food.
        :return: ``True`` if entry is affordable.
        :rtype: bool
        """
        return self.terrain.can_enter(strength, water, food)

    def add_item(self, item):
        """Add an item (or trader) onto this tile.

        :param Items.Items item: Item or :class:`Trader.Trader` to add.
        """
        if item is not None:
            self.items.append(item)

    def has_trader(self):
        """Return whether this tile contains a trader.

        :return: ``True`` if any :class:`Trader.Trader` is present.
        :rtype: bool
        """
        return any(isinstance(obj, Trader.Trader) for obj in self.items)

    def collect_items(self, player):
        """Transfer non-trader items to the player.

        Repeating items remain on the tile; one-time items are removed after
        collection.

        Scales resources based on tile cost and difficulty.
        """
        keep = []

        # Determine buffer based on difficulty
        buffer = 5
        if hasattr(self, 'difficulty') and self.difficulty in ["hard", "extreme"]:
            buffer = 2

        m_cost, w_cost, f_cost = self.get_costs()

        for item in self.items:
            # traders aren't collected here
            if isinstance(item, Trader.Trader):
                keep.append(item)
                continue

            # Peek at amount to avoid consuming it prematurely
            raw_amount = item.itemAmount
            if raw_amount > 0:
                # Scale amount if it's too low compared to costs
                scaled_amount = raw_amount
                if item.getType() == "water":
                    needed = w_cost + buffer
                    if scaled_amount < needed:
                        scaled_amount = needed
                elif item.getType() == "food":
                    needed = f_cost + buffer
                    if scaled_amount < needed:
                        scaled_amount = needed

                # Update item with scaled amount
                item.itemAmount = scaled_amount

                player.collect(item)

            # repeating items stay on tile
            if item.isRepeating and item.itemAmount > 0:
                keep.append(item)

        self.items = keep

    def __repr__(self):
        """Debug representation with terrain name and item count."""
        return f"Tile({self.terrain.name}, items={len(self.items)})"

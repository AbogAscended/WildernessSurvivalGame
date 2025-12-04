"""Player entity and resource management.

The :class:`Player` tracks resources (strength, water, food, gold), holds a
reference to the current :class:`src.Map.Map`, and provides helpers to collect
items and trade with a :class:`src.Trader.Trader`.
"""

# Prefer package-relative imports; fall back to absolute for script mode
try:  # pragma: no cover - import resolution shim
    from . import Items, Trader  # type: ignore
except Exception:  # pragma: no cover
    import Items  # type: ignore
    import Trader  # type: ignore

class Player:
    """Stateful player model used by the WSS.

    Attributes
    ----------
    maxFood, maxWater, maxStrength : int
        Maximum capacities for the three consumable resources.
    currentGold, currentWater, currentFood : int
        Current resource levels.
    currentStrength : int
        Current movement/strength points (separate from maxStrength).
    position : tuple[int, int]
        Current map coordinates ``(x, y)``.
    map_reference : src.Map.Map | None
        Reference to the attached map (set via :meth:`attach_map`).
    """
    maxFood = maxWater = maxStrength = 0
    currentGold = currentWater = currentFood = 0
    position = (0, 0)
    map_reference = None

    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reinitialize player stats and position.

        Sets all maximums to 100, initializes current resources and strength,
        and places the player at ``(0, 0)``.
        """
        self.maxFood = self.maxWater = self.maxStrength = 100
        self.currentGold = 20
        self.currentWater = 20
        self.currentFood = 20
        # track movement/strength points separately from max
        self.currentStrength = self.maxStrength
        self.position = (0, 0)

    def attach_map(self, new_map):
        """Attach the current map to this player.

        :param src.Map.Map new_map: Map instance to attach. This enables
            other components (Vision/Brain/Env) to access the world via the
            player.
        """
        self.map_reference = new_map
    
    def execute_trade(self,
                      trader,
                      option_index: int
                      ) -> bool:
        """Execute a specific trade option from the trader.

        :param Trader.Trader trader: The trader instance.
        :param int option_index: Index of the trade in trader's inventory (0-based).
        :return: ``True`` if trade successful, ``False`` otherwise.
        """
        inventory = trader.getInventory()
        if option_index < 0 or option_index >= len(inventory):
            return False

        proposal = inventory[option_index]
        # proposal: [trader_receives (what player pays), trader_offers]
        wants = proposal[0]  # [gold, water, food]

        if (self.currentGold >= wants[0] and
            self.currentWater >= wants[1] and
            self.currentFood >= wants[2]):

            self.transfer(proposal)
            return True
        return False

    def collect(self, item: Items):  # only for collecting items
        """Collect the provided non-trader item into player's inventory.

        Clamps water and food to their respective maximums. Gold has no max.

        :param Items.Items item: Item to collect (water/food/gold).
        """
        type, amount = item.getType(), item.getAmount()
        match type:
            case "water":
                self.currentWater = min(self.maxWater, self.currentWater + amount)
            case "food":
                self.currentFood = min(self.maxFood, self.currentFood + amount)
            case "gold":
                self.currentGold += amount

    def transfer(self, offer):  # only for bargaining with trader
        """Apply a barter offer result to player's resources.

        The offer uses the format ``[offering, receiving]`` with each side a
        triplet ``[gold, water, food]``. Resources are updated accordingly and
        clamped to valid ranges.

        :param list offer: ``[[g_out, w_out, f_out], [g_in, w_in, f_in]]``
            representing the final accepted offer.
        """
        self.currentGold += (offer[1][0] - offer[0][0])
        self.currentWater = max(0, min(self.maxWater, self.currentWater + (offer[1][1] - offer[0][1])))
        self.currentFood = max(0, min(self.maxFood, self.currentFood + (offer[1][2] - offer[0][2])))

    def getResources(self):
        """Return a human-readable summary of resources.

        :return: Text summary including gold, water, and food.
        :rtype: str
        """
        return "gold:" + str(self.currentGold) + " water: " + str(self.currentWater) + " food: " + str(self.currentFood)


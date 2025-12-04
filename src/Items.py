"""Items module.

Defines the generic item container used on tiles for resources and traders.

An item can be repeating (e.g., a stream that provides water every time the
player visits the tile) or one-time (e.g., a stash of gold that disappears
after collection).
"""

import random


class Items:
    """Generic item entity stored on a map tile.

    :param str type: Logical type of the item (``"water"``, ``"food"``,
        ``"gold"``, or ``"trader"`` where trader subclasses this base).
    :param bool repeat: Whether the item repeats (stays on the tile and can be
        collected again each visit).
    :param int amount: Quantity provided when collected. For repeating items,
        this is the per-visit amount. For non-repeating items, the remaining
        amount is set to zero after the first collection.

    :ivar str itemType: Item type identifier.
    :ivar bool isRepeating: Whether the item repeats.
    :ivar int itemAmount: Remaining amount (or per-visit amount if repeating).
    """

    itemType, isRepeating, itemAmount = 0, False, 0

    def __init__(self, type: str, repeat: bool, amount: int):
        self.itemType = type
        self.isRepeating = repeat
        self.itemAmount = amount

    def getAmount(self) -> int:
        """Return the collectible amount for this item and update state.

        :return: For repeating items, the configured amount (no state change).
            For non-repeating items, the remaining amount and sets it to zero
            thereafter.
        :rtype: int
        """
        if self.isRepeating:
            return self.itemAmount
        else:
            temp = self.itemAmount
            self.itemAmount = 0
            return temp

    def getType(self) -> str:
        """Get the logical type of the item.

        :return: Item type string (e.g., ``"water"``, ``"food"``, ``"gold"``,
            or ``"trader"``).
        :rtype: str
        """
        return self.itemType
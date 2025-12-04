"""Vision utilities for scanning nearby tiles.

The :class:`Vision` class maintains a radial ordering of relative offsets
within a configurable radius and provides helpers to locate the closest or
second-closest tiles that contain specific items.
"""

# import Map
# import Player


class Vision:
    """Vision system attached to a player for local scanning.

    Attributes
    ----------
    player_reference : Player | None
        The player to which this vision is attached (via :meth:`attach_player`).
    vision_radius : int
        Maximum Manhattan radius to scan around the player. Default is 4.
    closeList : list[tuple[int, int]]
        Ordered list of relative offsets starting from (0,0) and moving outward
        by increasing distance; used to iterate nearby tiles in priority order.
    """

    player_reference = None
    vision_radius = 4
    closeList = [(0, 0)]

    def setVisionRadius(self, vision_radius):
        """Set the scan radius and recompute the ordered offsets.

        :param int vision_radius: Maximum Manhattan distance to include.
        :return: None
        :rtype: None
        """
        self.vision_radius = vision_radius
        self.assembleCloseList()

    def assembleCloseList(self):
        """Recompute ``closeList`` in increasing distance order.

        Populates :attr:`closeList` with coordinate offsets starting at
        ``(0, 0)`` then expanding ring by ring to the configured radius.
        """
        self.closeList = []
        for i in range(0, self.vision_radius + 1):
            for k in sorted(range(i * -1, i + 1), key=abs):
                self.closeList.append((i - abs(k), k))

    def inBounds(self, pos, size):
        """Check whether an absolute position is within map bounds.

        :param tuple[int, int] pos: Absolute coordinates ``(x, y)``.
        :param tuple[int, int] size: Map size ``(width, height)``.
        :return: ``True`` if ``0 <= x < width`` and ``0 <= y < height``.
        :rtype: bool
        """
        return (
            (pos[0] >= 0 and pos[1] >= 0) and
            (pos[0] < size[0] and pos[1] < size[1])
        )

    def closestItem(self, itemType):
        """Find the closest tile containing an item of the given type.

        Scans tiles around the player in the precomputed priority order and
        returns the absolute position of the first tile that contains an item
        with ``getType() == itemType``.

        :param str itemType: Item type (e.g., ``"food"``, ``"water"``,
            ``"gold"``, or ``"trader"``).
        :return: Absolute coordinates ``(x, y)`` or ``None`` if not found.
        :rtype: tuple[int, int] | None
        """
        if self.player_reference is None or self.player_reference.map_reference is None:
            return None
        map_reference = self.player_reference.map_reference
        size = (map_reference.width, map_reference.height)
        for rel_pos in self.closeList:
            true_pos = (
                self.player_reference.position[0] + rel_pos[0],
                self.player_reference.position[1] + rel_pos[1],
            )
            if self.inBounds(true_pos, size):
                current_tile = map_reference.getTile(true_pos[0], true_pos[1])
                # search all items on the tile
                for itm in getattr(current_tile, "items", []):
                    try:
                        if itm.getType() == itemType:
                            return true_pos
                    except Exception:
                        continue
        return None

    def secondClosestItem(self, itemType):
        """Find the second closest tile containing a given item type.

        :param str itemType: Item type string to search for.
        :return: Absolute coordinates of the second match, or ``None``.
        :rtype: tuple[int, int] | None
        """
        if self.player_reference is None or self.player_reference.map_reference is None:
            return None
        map_reference = self.player_reference.map_reference
        size = (map_reference.width, map_reference.height)
        found_first = False
        for rel_pos in self.closeList:
            true_pos = (
                self.player_reference.position[0] + rel_pos[0],
                self.player_reference.position[1] + rel_pos[1],
            )
            if self.inBounds(true_pos, size):
                current_tile = map_reference.getTile(true_pos[0], true_pos[1])
                for itm in getattr(current_tile, "items", []):
                    try:
                        if itm.getType() == itemType:
                            if not found_first:
                                found_first = True
                            else:
                                return true_pos
                    except Exception:
                        continue
        return None

    def attach_player(self, player):
        """Attach a player to this vision instance.

        :param Player player: The player to be used as scan origin.
        :return: None
        :rtype: None
        """
        self.player_reference = player

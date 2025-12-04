"""Brain (decision policy) base class.

This module defines a minimal :class:`Brain` shell that wires together a
player and a vision component. In a richer implementation you would subclass
``Brain`` and implement a decision method (e.g., ``makeMove``) that consults
vision and player state to choose an action.
"""

try:
    from . import Player, Vision  # type: ignore
except Exception:
    import Player  # type: ignore
    import Vision  # type: ignore

class Brain:
    """Base brain with references to a player and a vision system.

    Attributes
    ----------
    player_reference : Player | None
        The player controlled by this brain.
    vision_reference : Vision | None
        Vision helper used to query nearby tiles/items.
    """

    player_reference = None
    vision_reference = None

    def attach_player(self, player):
        """Attach a player to be controlled by this brain.

        :param Player.Player player: Player instance.
        :return: None
        :rtype: None
        """
        self.player_reference = player

    def attach_vision(self, vision):
        """Attach a vision helper used by the brain for perception.

        :param Vision.Vision vision: Vision instance.
        :return: None
        :rtype: None
        """
        self.vision_reference = vision

"""Map generation and item placement.

This module defines the :class:`Map` container responsible for holding the grid
of :class:`src.Tiles.Tile` objects, generating terrain and items according to a
given difficulty, and wiring the player reference.

Key features
------------
- Difficulty-aware terrain and item spawning (uses :mod:`src.Difficulty` when
  available).
- Reproducible randomness via a dedicated :class:`random.Random` RNG instance
  that may be passed in.
"""

import random

# Prefer package-relative imports; fall back to absolute when running modules directly
try:  # pragma: no cover - import resolution shim
    from . import Terrain, Tiles, Items, Trader, Player, Difficulty  # type: ignore
except Exception:  # pragma: no cover
    try:
        import Terrain  # type: ignore
        import Tiles  # type: ignore
        import Items  # type: ignore
        import Trader  # type: ignore
        import Player  # type: ignore
    except Exception as _e:  # type: ignore
        raise
    try:
        import Difficulty  # type: ignore
    except Exception:
        Difficulty = None  # type: ignore

# TODO: adjust as needed, just example for now
ITEM_PROBABILITY_TABLE = [0.5, 0.1, 0.01]


class Map:
    """2D grid of tiles with difficulty-scaled content.

    Parameters
    ----------
    width : int
        Number of columns in the map (west→east).
    height : int
        Number of rows in the map (north→south).
    player : src.Player.Player
        Player instance to attach to this map (used by vision/collection).
    difficulty : str, optional
        One of ``"easy"``, ``"medium"`, ``"hard"``, ``"extreme"``. The alias
        ``"normal"`` maps to ``"medium"``.
    rng : random.Random, optional
        RNG to use for deterministic generation. If omitted, a new RNG is
        created.

    Attributes
    ----------
    width, height : int
        Map dimensions.
    player : Player | None
        Attached player reference (if any).
    difficulty : str
        Canonical difficulty label.
    tile_map : list[list[Tile]]
        2D list of tiles of shape ``[width][height]``.
    _rng : random.Random
        RNG used for all stochastic generation.
    """
    width = 0
    height = 0
    player = None
    difficulty = "normal"
    tile_map = None
    _rng = None

    def reset(self,
              width=0,
              height=0,
              player=None,
              difficulty=None,
              rng: random.Random | None = None
              ):
        """(Re)initialize the map.

        :param int width: Optional new width; if > 0, overrides existing.
        :param int height: Optional new height; if > 0, overrides existing.
        :param Player player: Optional new player to attach.
        :param str difficulty: Optional difficulty label to (re)generate map.
            Accepted values: ``easy|medium|hard|extreme``; ``normal`` aliases
            to ``medium``. If not provided, keeps the current setting.
        :param random.Random rng: Optional RNG to use for this and subsequent
            generations.
        :return: None
        :rtype: None
        """
        regenerate = False
        if width and width > 0:
            regenerate = True
            self.width = width
        if height and height > 0:
            regenerate = True
            self.height = height
        if player is not None:
            self.player = player
        if difficulty is not None:
            regenerate = True
            if Difficulty is not None:
                self.difficulty = Difficulty.canonicalize(difficulty)
            else:
                self.difficulty = difficulty
        # RNG wiring
        if rng is not None:
            self._rng = rng
        if self._rng is None:
            self._rng = random.Random()
        if regenerate:
            self.tile_map = []
            # Item spawn config by difficulty
            if Difficulty is not None:
                cfg = Difficulty.get_item_config(self.difficulty)
            else:
                cfg = {
                    "trader_prob": 0.05,
                    "water_prob": 0.08,
                    "food_prob": 0.06,
                    "gold_prob": 0.04,
                    "water_repeat_prob": 0.5,
                    "food_repeat_prob": 0.35,
                    "water_amount": (2, 5),
                    "food_amount": (2, 5),
                    "gold_amount": (3, 6),
                }
            for x in range(self.width):
                current_row = []
                for y in range(self.height):
                    current_tile = Tiles.Tile(terrain=None, difficulty=self.difficulty)
                    # populate items/traders according to difficulty-scaled probs
                    r = self._rng.random()
                    if r < float(cfg.get("trader_prob", 0.05)):
                        # trader spawns exclusively
                        current_tile.add_item(Trader.Trader(self.difficulty, tile_costs=current_tile.get_costs(), rng=self._rng))
                    else:
                        r2 = self._rng.random()
                        wp = float(cfg.get("water_prob", 0.08))
                        fp = float(cfg.get("food_prob", 0.06))
                        gp = float(cfg.get("gold_prob", 0.04))
                        # cumulative bands
                        if r2 < wp:
                            lo, hi = cfg.get("water_amount", (2, 5))
                            amt = self._rng.randint(int(lo), int(hi))
                            rep = self._rng.random() < float(cfg.get("water_repeat_prob", 0.5))
                            current_tile.add_item(Items.Items("water", rep, amt))
                        elif r2 < wp + fp:
                            lo, hi = cfg.get("food_amount", (2, 5))
                            amt = self._rng.randint(int(lo), int(hi))
                            rep = self._rng.random() < float(cfg.get("food_repeat_prob", 0.35))
                            current_tile.add_item(Items.Items("food", rep, amt))
                        elif r2 < wp + fp + gp:
                            lo, hi = cfg.get("gold_amount", (3, 6))
                            amt = self._rng.randint(int(lo), int(hi))
                            current_tile.add_item(Items.Items("gold", False, amt))
                    current_row.append(current_tile)
                self.tile_map.append(current_row)

    def __init__(self, width, height, player, difficulty="normal", rng: random.Random | None = None):
        """Create a map and immediately generate its tiles.

        :param int width: Number of columns.
        :param int height: Number of rows.
        :param Player player: Player to attach.
        :param str difficulty: Difficulty label (``easy|medium|hard|extreme``;
            ``normal`` aliases to ``medium``).
        :param random.Random rng: Optional RNG to use.
        """
        self._rng = rng or random.Random()
        self.reset(width, height, player, difficulty, rng=self._rng)

    def getTile(self, x, y):
        """Return the tile at coordinates ``(x, y)``.

        :param int x: Column index (0-based from west).
        :param int y: Row index (0-based from north).
        :return: The :class:`src.Tiles.Tile` at the given coordinates.
        :rtype: Tiles.Tile
        """
        return self.tile_map[x][y]

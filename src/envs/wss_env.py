"""Gymnasium environment wrapper for the Wilderness Survival System (WSS).

This module exposes :class:`WildernessSurvivalEnv`, a discrete-action RL
environment that wraps the WSS core (map, player, terrain, items). Rendering
is toggleable (ASCII or RGB array). Rewards are intentionally neutral by
default; a dedicated section in :meth:`WildernessSurvivalEnv.step` indicates
where to add custom shaping.
"""

import math
import random
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Local imports
from src import Terrain
from src.Map import Map
from src.Player import Player
from src import Trader  # Added for automatic trading logic
try:
    from src import Difficulty  # type: ignore
except Exception:  # pragma: no cover
    Difficulty = None  # type: ignore


class WildernessSurvivalEnv(gym.Env):
    """Gymnasium environment wrapper for the Wilderness Survival System (WSS).

    Key notes
    ---------
    - Discrete actions (``0..8``): Stay, N, S, E, W, NE, NW, SE, SW.
    - Observation: flat vector
        ``[ one-hot terrain window , normalized stats , normalized position ]``
        where stats = ``[strength/max, water/max, food/max, gold_norm]`` and
        position = ``[x/(W-1), y/(H-1)]``.
    - Rendering: Toggleable. ``human`` prints ASCII; ``rgb_array`` returns an
      RGB image. ``None`` disables rendering for speed.
    - Rewards: By default 0. See the clearly marked section in
      :meth:`step` to customize.

    Episode termination
    -------------------
    - Reach the east edge (``x == width-1``), or
    - Any resource depleted (``strength<=0`` or ``water<=0`` or ``food<=0``)

    Truncation
    ----------
    - Optional ``max_steps`` cap
    """

    metadata = {"render_modes": [None, "human", "rgb_array", "pygame"], "render_fps": 4}

    ACTIONS = (
        (0, 0),   # 0 Stay
        (0, -1),  # 1 North
        (0, 1),   # 2 South
        (1, 0),   # 3 East
        (-1, 0),  # 4 West
        (1, -1),  # 5 NorthEast
        (-1, -1), # 6 NorthWest
        (1, 1),   # 7 SouthEast
        (-1, 1),  # 8 SouthWest
    )

    TERRAIN_TO_IDX = {t.name: i for i, t in enumerate(Terrain.TERRAIN_TYPES)}
    IDX_TO_TERRAIN = {i: t.name for i, t in enumerate(Terrain.TERRAIN_TYPES)}

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        difficulty: str = "normal",
        render_mode: Optional[str] = None,
        window_radius: int = 2,
        max_steps: Optional[int] = None,
        # ASCII render options
        render_style: str = "nethack",
        show_legend: bool = True,
        ascii_colors: bool = False,
        clear_screen: bool = False,
        # Pygame render options
        pygame_cell_size: int = 28,
        pygame_show_grid: bool = True,
        pygame_legend_width: int = 240,
        pygame_fps: int = 30,
        # Reward shaping
        gamma: float = 0.99,
    ) -> None:
        """Construct the WSS Gymnasium environment.

        :param int|None width: Map width (columns). If None, determined by difficulty.
        :param int|None height: Map height (rows). If None, determined by difficulty.
        :param str difficulty: Difficulty label (``easy|medium|hard|extreme``;
            ``normal`` aliases to ``medium``).
        :param str|None render_mode: One of ``None``, ``"human"`` (ASCII),
            or ``"rgb_array"``.
        :param int window_radius: Radius ``r`` for the local terrain window;
            the one-hot window covers a ``(2r+1) x (2r+1)`` area.
        :param int|None max_steps: Optional episode step cap (truncation).
        :param float gamma: Discount factor for potential-based reward shaping.
            Should match the RL algorithm's gamma.
        """
        super().__init__()
        assert render_mode in self.metadata["render_modes"], (
            f"render_mode must be one of {self.metadata['render_modes']}"
        )
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.gamma = gamma
        
        # Canonicalize difficulty
        self.difficulty = Difficulty.canonicalize(difficulty) if Difficulty else difficulty

        # Determine map size if not specified
        self._dynamic_size = (width is None or height is None)
        if self._dynamic_size:
            if Difficulty:
                w_min, w_max = Difficulty.MAP_SIZES.get(self.difficulty, (20, 20))
                side = random.randint(w_min, w_max)
                self.width = side if width is None else width
                self.height = side if height is None else height
            else:
                self.width = 20 if width is None else width
                self.height = 10 if height is None else height
        else:
            self.width = width
            self.height = height

        self.window_radius = int(window_radius)
        self.max_steps = max_steps
        # ASCII rendering options (used when render_mode == "human")
        self.render_style = render_style
        self.show_legend = bool(show_legend)
        self.ascii_colors = bool(ascii_colors)
        self.clear_screen = bool(clear_screen)
        # Pygame renderer configuration
        self._pg_renderer = None  # lazy init
        self._pg_cell_size = int(pygame_cell_size)
        self._pg_show_grid = bool(pygame_show_grid)
        self._pg_legend_w = int(pygame_legend_width)
        self._pg_fps = int(pygame_fps)

        # Core game objects
        self.player: Optional[Player] = None
        self.map: Optional[Map] = None

        # Action space: 13 discrete actions (9 movement + 4 trade)
        self.action_space = spaces.Discrete(13)

        # Observation space: Dict with local matrices and status vectors
        self.observation_space = spaces.Dict({
            "terrain": spaces.Box(low=0, high=5, shape=(11, 11), dtype=np.int32),
            "items": spaces.Box(low=0, high=4, shape=(11, 11), dtype=np.int32),
            "status": spaces.Box(low=0, high=10000, shape=(4,), dtype=np.int32),
            "trader": spaces.Box(low=-1000, high=1000, shape=(4, 4), dtype=np.int32),
            "prev_action": spaces.Box(low=0, high=12, shape=(1,), dtype=np.int32),
        })

        # Internal counters
        self._steps = 0
        self.prev_action = 0

    @property
    def renderer(self):
        """Expose the renderer instance (if any) for external overlays."""
        return self._pg_renderer

    # --------------- Gymnasium API ---------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment to start a new episode.

        Seeds numpy and Python RNGs when a seed is provided, creates a fresh
        player and map, attaches references, and places the player at the west
        edge centered vertically.

        Also initializes auxiliary state used by the reward shaper
        (``_best_x``, ``_gold_prev``, and ``_h_before``).

        :param int|None seed: Optional seed for reproducibility.
        :param dict|None options: Reserved for Gymnasium compatibility.
        :return: Tuple of initial observation and info dict.
        :rtype: tuple[dict, dict]
        """
        super().reset(seed=seed)
        # Seed numpy & python RNGs for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        rng = random.Random(seed) if seed is not None else random.Random()

        # Initialize player and map
        self.player = Player()
        self.player.reset()
        # Canonicalize difficulty to support alias "normal" -> "medium"
        diff = Difficulty.canonicalize(self.difficulty) if Difficulty is not None else self.difficulty
        self.difficulty = diff

        # Update map dimensions if dynamic sizing is enabled
        if getattr(self, "_dynamic_size", False) and Difficulty is not None:
            w_min, w_max = Difficulty.MAP_SIZES.get(self.difficulty, (20, 20))
            # Only re-roll if we don't have fixed constraints
            side = rng.randint(w_min, w_max)
            self.width = side
            self.height = side

        self.map = Map(self.width, self.height, self.player, diff, rng=rng)
        self.player.attach_map(self.map)

        # Start at west edge, center row
        start_y = self.height // 2
        self.player.position = (0, start_y)
        self._steps = 0

        # Initialize auxiliary reward tracking
        self._best_x = 0
        self._gold_prev = self.player.currentGold
        self._h_before = min(
            self.player.currentWater / max(1, self.player.maxWater),
            self.player.currentFood / max(1, self.player.maxFood),
            getattr(self.player, "currentStrength", 0) / max(1, self.player.maxStrength),
        )
        # Per-turn usage tracker (movement, water, food)
        self._last_usage = {"move": 0, "water": 0, "food": 0}
        self.prev_action = 0

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_ascii()
        elif self.render_mode == "pygame":
            self._render_pygame()
        return obs, info

    def step(self, action: int):
        """Apply an action and advance the environment by one step.

        Actions are discrete integers in ``[0, 8]`` mapping to cardinal and
        diagonal moves with ``0`` meaning "stay" (rest). Entering a tile pays
        the terrain costs (strength, water, food). Staying restores 2 strength
        and pays half water/food costs of the current tile.

        Termination occurs upon reaching the east edge or depleting any
        resource. Truncation can occur at ``max_steps``.

        Reward shaping (wired-in)
        -------------------------
        The reward aligns with the objective "reach the east edge as fast as
        possible" using potential-based shaping and event bonuses:
        - Eastward progress: potential on x-position; only pays for net east
          movement (no penalty for waiting or moving sideways/backwards).
        - Survival improvement: potential on the minimum of normalized
          water/food; rewards net improvements (e.g., collecting/trading).
        - Discovery bonus: small bonus when surpassing the best x seen so far
          in the episode.
        - Goal bonus / Death penalty: +1 on success, -1 on failure.
        - Tiny gold utility bonus for increases in gold.
        Rewards are mildly scaled by difficulty hardness and clipped to
        ``[-1, 1]`` for PPO stability.

        :param int action: Discrete action ``0..8``.
        :return: ``(obs, reward, terminated, truncated, info)`` per Gymnasium.
        :rtype: tuple[numpy.ndarray, float, bool, bool, dict]
        """
        assert self.player is not None and self.map is not None
        self._steps += 1

        action = int(action)
        px, py = self.player.position
        # Capture survival potential BEFORE applying any dynamics
        h_before = min(
            self.player.currentWater / max(1, self.player.maxWater),
            self.player.currentFood / max(1, self.player.maxFood),
            getattr(self.player, "currentStrength", 0) / max(1, self.player.maxStrength),
        )

        # Resolve Action
        if action >= 9:
            # Trade Action (9-12)
            dx, dy = 0, 0
            trade_idx = action - 9
            # Execute trade (after capturing h_before, so gains are rewarded)
            if 0 <= px < self.width and 0 <= py < self.height:
                tile = self.map.getTile(px, py)
                trader = next((i for i in getattr(tile, "items", []) if isinstance(i, Trader.Trader)), None)
                if trader:
                    self.player.execute_trade(trader, trade_idx)
        else:
            # Movement Action (0-8)
            dx, dy = self.ACTIONS[action]

        nx, ny = px + dx, py + dy

        terminated = False
        truncated = False

        moved = False
        # Track per-turn resource usage (costs paid). Positive numbers indicate consumption.
        move_used = 0
        water_used = 0
        food_used = 0
        if dx == 0 and dy == 0:
            # Stay still: regain 2 units of movement (strength), half water/food costs of current tile
            tile = self.map.getTile(px, py)
            m_cost, w_cost, f_cost = tile.get_costs()

            w_half = math.ceil(w_cost / 2)
            f_half = math.ceil(f_cost / 2)
            self.player.currentWater = max(0, self.player.currentWater - w_half)
            self.player.currentFood = max(0, self.player.currentFood - f_half)
            water_used += w_half
            food_used += f_half
            self.player.currentStrength = max(0, min(self.player.maxStrength, self.player.currentStrength + 5))
            move_used += 0  # regained strength; do not count as usage
        else:
            # Move attempt
            if 0 <= nx < self.width and 0 <= ny < self.height:
                target_tile = self.map.getTile(nx, ny)
                m_cost, w_cost, f_cost = target_tile.get_costs()

                # Ensure player has currentStrength attribute (not in original skeleton)
                if not hasattr(self.player, "currentStrength"):
                    # initialize if missing
                    self.player.currentStrength = self.player.maxStrength

                can_enter = target_tile.is_passable(
                    strength=getattr(self.player, "currentStrength", self.player.maxStrength),
                    water=self.player.currentWater,
                    food=self.player.currentFood,
                )
                if can_enter:
                    self.player.position = (nx, ny)
                    # Apply costs on enter
                    self.player.currentStrength = max(0, self.player.currentStrength - m_cost)
                    self.player.currentWater = max(0, self.player.currentWater - w_cost)
                    self.player.currentFood = max(0, self.player.currentFood - f_cost)
                    move_used += m_cost
                    water_used += w_cost
                    food_used += f_cost
                    moved = True
                    # Collect any items
                    target_tile.collect_items(self.player)
                else:
                    # Invalid move: we will treat as a "wait" with half-costs of current tile
                    tile = self.map.getTile(px, py)
                    m_cost, w_cost, f_cost = tile.get_costs()
                    w_half = math.ceil(w_cost / 2)
                    f_half = math.ceil(f_cost / 2)
                    self.player.currentWater = max(0, self.player.currentWater - w_half)
                    self.player.currentFood = max(0, self.player.currentFood - f_half)
                    self.player.currentStrength = max(0, min(self.player.maxStrength, self.player.currentStrength + 5))
                    water_used += w_half
                    food_used += f_half
            else:
                # Out of bounds -> treat as wait
                tile = self.map.getTile(px, py)
                m_cost, w_cost, f_cost = tile.get_costs()
                w_half = math.ceil(w_cost / 2)
                f_half = math.ceil(f_cost / 2)
                self.player.currentWater = max(0, self.player.currentWater - w_half)
                self.player.currentFood = max(0, self.player.currentFood - f_half)
                self.player.currentStrength = max(0, min(self.player.maxStrength, self.player.currentStrength + 5))
                water_used += w_half
                food_used += f_half

        # Check termination conditions
        if self.player.position[0] == self.width - 1:
            terminated = True
        if (
            getattr(self.player, "currentStrength", 0) <= 0
            or self.player.currentWater <= 0
            or self.player.currentFood <= 0
        ):
            terminated = True

        if self.max_steps is not None and self._steps >= self.max_steps:
            truncated = True

        # Weights
        w_goal = 1.0
        w_death = 0.5  # Reduced from 1.0 to make death less dominant vs progress
        w_east = 1.0   # Increased from 0.5 to encourage forward progress more
        w_surv = 0.5   # Increased from 0.3 to emphasize resource maintenance
        w_disc = 0.5   # Increased from 0.2 to reward exploration
        w_gold = 0.000 # Reduced from 0.01 to prevent infinite reward loops via trading
        w_time = 0.000  # Reduced from 0.01 to allow more exploration time

        # Read hardness if available (0..1). Scale rewards mildly by difficulty.
        try:
            hardness = Difficulty.get_hardness(self.difficulty) if Difficulty is not None else 0.0
        except Exception:
            hardness = 0.0

        # Optional difficulty scaling (mild)
        w_goal *= (1.0 + 0.5 * hardness)
        w_east *= (1.0 + 0.25 * hardness)

        # Potentials before and after transition
        Wm1 = max(1, self.width - 1)
        # East potential
        phi_e_before = px / Wm1
        phi_e_after = self.player.position[0] / Wm1
        # Survival potential h_before captured earlier

        # Track prior resources for deltas (gold only).
        prev_gold = getattr(self, "_gold_prev", self.player.currentGold)

        # Discovery/frontier state
        best_x = getattr(self, "_best_x", 0)

        # Now compute shaped rewards
        gamma_shaping = self.gamma
        r_east = w_east * (gamma_shaping * phi_e_after - phi_e_before)

        # Survival improvement uses current h minus stored h_before
        h_now = min(
            self.player.currentWater / max(1, self.player.maxWater),
            self.player.currentFood / max(1, self.player.maxFood),
            getattr(self.player, "currentStrength", 0) / max(1, self.player.maxStrength),
        )
        r_surv = w_surv * (gamma_shaping * h_now - h_before)

        # Discovery: reward surpassing prior best x (only positive)
        r_disc = 0.0
        if self.player.position[0] > best_x:
            r_disc = w_disc * (self.player.position[0] - best_x) / Wm1
            best_x = self.player.position[0]

        # Gold utility (small, only for increases)
        r_gold = w_gold * (self.player.currentGold - prev_gold)
        if action >= 9:
            r_gold = 0.0
        
        # Time penalty
        r_time = -w_time

        # Terminals
        r_goal = w_goal if (terminated and self.player.position[0] == self.width - 1) else 0.0
        r_death = -w_death if (
                terminated and (
                getattr(self.player, "currentStrength", 0) <= 0
                or self.player.currentWater <= 0 or self.player.currentFood <= 0
        )
        ) else 0.0

        reward = r_east + r_surv + r_disc + r_gold + r_goal + r_death + r_time

        # Persist auxiliary state for next step
        self._best_x = best_x
        self._gold_prev = self.player.currentGold
        self._h_before = h_now
        # Save last turn usage (non-negative values)
        try:
            self._last_usage = {
                "move": int(max(0, move_used)),
                "water": int(max(0, water_used)),
                "food": int(max(0, food_used)),
            }
        except Exception:
            pass

        # Safety: clip extreme values (optional)
        reward = float(np.clip(reward, -1.0, 1.0))

        self.prev_action = action
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_ascii()
        elif self.render_mode == "rgb_array":
            # The Gymnasium API expects `render()` to be called separately, but we allow it here too
            pass
        elif self.render_mode == "pygame":
            self._render_pygame()

        return obs, reward, terminated, truncated, info

    # --------------- Helpers ---------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build the dictionary observation with local view and status.

        - terrain: 11x11 grid of terrain types (0=Fog, 1..5=Types)
        - items: 11x11 grid of items (0=None, 1=Trader, 2=Water, 3=Food, 4=Gold)
        - status: [Food, Water, Strength, Gold]
        - trader: 4x4 matrix of trade options (relative resource changes)
        """
        assert self.map is not None and self.player is not None
        px, py = self.player.position

        # 1. Prepare matrices
        terrain_mat = np.zeros((11, 11), dtype=np.int32)
        items_mat = np.zeros((11, 11), dtype=np.int32)

        # Vision radius for fog
        radius = 3
        if Difficulty:
            radius = Difficulty.VISION_RADII.get(self.difficulty, 3)

        # Populate 11x11 window (centered at 5,5); relative range -5 to +5
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                # Fog check (Chebyshev distance)
                dist = max(abs(dx), abs(dy))
                if dist > radius:
                    continue  # Leave as 0 (Fog)

                mx, my = px + dx, py + dy
                if 0 <= mx < self.width and 0 <= my < self.height:
                    tile = self.map.getTile(mx, my)

                    # Terrain mapping: Plains=1, Forest=2, Swamp=3, Mountain=4, Desert=5
                    t_name = tile.terrain.name
                    t_val = 0
                    if t_name == "plains": t_val = 1
                    elif t_name == "forest": t_val = 2
                    elif t_name == "swamp": t_val = 3
                    elif t_name == "mountain": t_val = 4
                    elif t_name == "desert": t_val = 5
                    terrain_mat[dy + 5, dx + 5] = t_val

                    # Item mapping: Trader=1, Water=2, Food=3, Gold=4 (Priority order)
                    i_val = 0
                    items = getattr(tile, "items", [])
                    has_trader = any(getattr(i, "itemType", "") == "trader" for i in items)
                    has_gold = any(getattr(i, "itemType", "") == "gold" for i in items)
                    has_water = any(getattr(i, "itemType", "") == "water" for i in items)
                    has_food = any(getattr(i, "itemType", "") == "food" for i in items)

                    if has_trader: i_val = 1
                    elif has_gold: i_val = 4
                    elif has_water: i_val = 2
                    elif has_food: i_val = 3

                    items_mat[dy + 5, dx + 5] = i_val

        # 2. Status Vector: [Food, Water, Strength, Gold]
        status_vec = np.array([
            self.player.currentFood,
            self.player.currentWater,
            getattr(self.player, "currentStrength", 0),
            self.player.currentGold
        ], dtype=np.int32)

        # 3. Trader Matrix
        trader_mat = np.zeros((4, 4), dtype=np.int32)
        tile = self.map.getTile(px, py)
        trader = next((i for i in getattr(tile, "items", []) if isinstance(i, Trader.Trader)), None)

        if trader:
            inventory = trader.getInventory()
            for i, proposal in enumerate(inventory):
                if i >= 4: break
                # proposal: [wants, offers] -> [Gold, Water, Food]
                wants = proposal[0]
                offers = proposal[1]

                # Calculate deltas: Offer - Want (Food, Water, Strength, Gold)
                d_food = offers[2] - wants[2]
                d_water = offers[1] - wants[1]
                d_strength = 0
                d_gold = offers[0] - wants[0]

                trader_mat[i] = [d_food, d_water, d_strength, d_gold]

        return {
            "terrain": terrain_mat,
            "items": items_mat,
            "status": status_vec,
            "trader": trader_mat,
            "prev_action": np.array([self.prev_action], dtype=np.int32)
        }

    def _get_info(self) -> Dict[str, Any]:
        """Return an info dictionary for diagnostics.

        Includes position, resources, difficulty label, numeric hardness, goal
        flag, and step count.

        :return: Info dict with diagnostic fields.
        :rtype: dict
        """
        assert self.player is not None
        # Difficulty diagnostics for RL/analysis
        hardness = None
        if Difficulty is not None:
            try:
                hardness = Difficulty.get_hardness(self.difficulty)
            except Exception:
                hardness = None
        # Tile diagnostics
        terrain_name = None
        tile_items: list[str] = []
        has_trader = False
        tile_costs = None
        try:
            assert self.map is not None
            tx, ty = self.player.position
            tile = self.map.getTile(tx, ty)
            terrain_name = getattr(tile.terrain, "name", None)
            try:
                m_c, w_c, f_c = tile.get_costs()
                tile_costs = {"move": int(m_c), "water": int(w_c), "food": int(f_c)}
            except Exception:
                tile_costs = None
            for itm in getattr(tile, "items", []):
                t = getattr(itm, "itemType", None)
                if t == "trader":
                    has_trader = True
                elif t in ("water", "food", "gold"):
                    tile_items.append(t)
        except Exception:
            pass
        # Last usage fallback
        last_usage = getattr(self, "_last_usage", None)
        info = {
            "position": tuple(self.player.position),
            "resources": {
                "strength": getattr(self.player, "currentStrength", self.player.maxStrength),
                "water": self.player.currentWater,
                "food": self.player.currentFood,
                "gold": self.player.currentGold,
                # maxima for HUDs
                "max_strength": getattr(self.player, "maxStrength", 100),
                "max_water": getattr(self.player, "maxWater", 100),
                "max_food": getattr(self.player, "maxFood", 100),
            },
            "difficulty": self.difficulty,
            "hardness": hardness,
            "reached_goal": self.player.position[0] == self.width - 1,
            "step": self._steps,
            "terrain": terrain_name,
            "tile_items": tile_items,
            "tile_has_trader": has_trader,
            "tile_costs": tile_costs,
            "last_usage": last_usage,
        }
        return info

    # --------------- Rendering ---------------
    def render(self):
        """Render the environment according to ``render_mode``.

        :return: ``None`` for ``human`` mode; RGB array for ``rgb_array``;
            otherwise ``None``.
        :rtype: numpy.ndarray | None
        """
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            self._render_ascii()
            return None
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "pygame":
            self._render_pygame()
            return None

    def _render_ascii(self):
        """Print a NetHack-style ASCII representation of the map with legend.

        Symbols:
        - Player: ``@``
        - Trader: ``R``
        - Items: ``$`` (gold), ``w`` (water), ``%`` (food)
        - Terrain: ``.`` (plains), ``T`` (forest), ``~`` (swamp), ``^`` (mountain), ``:`` (desert)
        """
        assert self.map is not None and self.player is not None

        # Optional clear screen in compatible terminals
        if self.clear_screen:
            print("\x1b[2J\x1b[H", end="")

        # Terrain glyphs
        terrain_glyph = {
            "plains": ".",
            "forest": "T",
            "swamp": "~",
            "mountain": "^",
            "desert": ":",
        }
        # ANSI colors (guarded by ascii_colors)
        color = {
            ".": "\x1b[38;5;250m",  # light gray
            "T": "\x1b[38;5;28m",   # green
            "~": "\x1b[38;5;31m",   # blue-ish
            "^": "\x1b[38;5;244m",  # gray
            ":": "\x1b[38;5;179m",  # sandy
            "$": "\x1b[33m",        # yellow
            "%": "\x1b[38;5;214m",  # orange
            "w": "\x1b[36m",        # cyan
            "R": "\x1b[35m",        # magenta
            "@": "\x1b[31m",        # red
        }
        reset = "\x1b[0m"

        # Determine radius based on difficulty
        radius = Difficulty.VISION_RADII.get(self.difficulty, 3) if Difficulty else 3
        px, py = self.player.position

        def glyph_for(x: int, y: int) -> str:
            # Fog of war check
            dist = max(abs(x - px), abs(y - py))
            if dist > radius:
                return " "

            # Overlay priority: player > trader > item > terrain
            if (x, y) == tuple(self.player.position):
                return "@"
            tile = self.map.getTile(x, y)
            # Trader present?
            if any(getattr(itm, "itemType", "") == "trader" for itm in getattr(tile, "items", [])):
                return "R"
            # Items
            has_gold = False
            has_water = False
            has_food = False
            for itm in getattr(tile, "items", []):
                t = getattr(itm, "itemType", "")
                if t == "gold":
                    has_gold = True
                elif t == "water":
                    has_water = True
                elif t == "food":
                    has_food = True
            if has_gold:
                return "$"
            if has_water:
                return "w"
            if has_food:
                return "%"
            # Terrain fallback
            return terrain_glyph.get(self.map.getTile(x, y).terrain.name, "?")

        # Build map rows (restricted to vision radius)
        map_rows: list[str] = []
        
        min_y = max(0, py - radius)
        max_y = min(self.height - 1, py + radius)
        min_x = max(0, px - radius)
        max_x = min(self.width - 1, px + radius)

        for y in range(min_y, max_y + 1):
            row_chars: list[str] = []
            for x in range(min_x, max_x + 1):
                ch = glyph_for(x, y)
                if self.ascii_colors and ch in color:
                    row_chars.append(color[ch] + ch + reset)
                else:
                    row_chars.append(ch)
            map_rows.append("".join(row_chars))

        # Build legend (top-right)
        legend_lines: list[str] = []
        if self.show_legend:
            header = f"Legend  diff={self.difficulty}  step={self._steps}"
            legend_lines.append(header)
            # Controls section at the top for clarity
            legend_lines.append("")
            legend_lines.append("Controls:")
            legend_lines.append("  h/j/k/l or WASD = Move")
            legend_lines.append("  y/u/b/n or q/e/z/c = Diagonals")
            legend_lines.append("  . or Space = Wait/Rest")
            legend_lines.append("  Q = Quit")
            legend_lines.append("")
            # Status section (resources and position)
            try:
                s_cur = getattr(self.player, "currentStrength", self.player.maxStrength)
                s_max = getattr(self.player, "maxStrength", 100)
                w_cur = self.player.currentWater
                w_max = getattr(self.player, "maxWater", 100)
                f_cur = self.player.currentFood
                f_max = getattr(self.player, "maxFood", 100)
                g_cur = self.player.currentGold
                px, py = self.player.position
                def bar(cur: int, mx: int, width: int = 12) -> str:
                    mx = max(1, int(mx))
                    frac = max(0.0, min(1.0, float(cur) / float(mx)))
                    filled = int(round(frac * width))
                    return "#" * filled + "." * (width - filled)
                legend_lines.append("Status:")
                legend_lines.append(f"  STR {s_cur}/{s_max} [{bar(s_cur, s_max)}]")
                legend_lines.append(f"  WAT {w_cur}/{w_max} [{bar(w_cur, w_max)}]")
                legend_lines.append(f"  FOD {f_cur}/{f_max} [{bar(f_cur, f_max)}]")
                legend_lines.append(f"  GLD {g_cur}")
                legend_lines.append(f"  Pos ({px},{py}) / ({self.width},{self.height})")
                # Tile info
                try:
                    tile = self.map.getTile(px, py)
                    tname = getattr(tile.terrain, "name", "?")
                    items_here = []
                    has_trader = False
                    for itm in getattr(tile, "items", []):
                        t = getattr(itm, "itemType", None)
                        if t == "trader":
                            has_trader = True
                        elif t in ("water", "food", "gold"):
                            items_here.append(t)
                    legend_lines.append(f"  Terrain: {tname}")
                    if has_trader or items_here:
                        items_txt = ("Trader "+("+ "+", ".join(items_here) if items_here else "")).strip()
                        if not has_trader:
                            items_txt = ", ".join(items_here)
                        legend_lines.append(f"  Here: {items_txt}")
                    # Show tile costs and last usage if available
                    try:
                        m_c, w_c, f_c = tile.get_costs()
                        legend_lines.append(f"  Tile cost M/W/F: {m_c}/{w_c}/{f_c}")
                    except Exception:
                        pass
                    lu = getattr(self, "_last_usage", None)
                    if isinstance(lu, dict):
                        legend_lines.append(
                            f"  Last use W/F: {int(lu.get('water',0))}/{int(lu.get('food',0))}"
                        )
                except Exception:
                    pass
                legend_lines.append("")
            except Exception:
                pass
            items = [
                ("@", "Player"),
                ("R", "Trader"),
                ("$", "Gold"),
                ("w", "Water"),
                ("%", "Food"),
                (".", "Plains"),
                ("T", "Forest"),
                ("~", "Swamp"),
                ("^", "Mountain"),
                (":", "Desert"),
            ]
            for sym, desc in items:
                if self.ascii_colors and sym in color:
                    legend_lines.append(f" {color[sym]}{sym}{reset} - {desc}")
                else:
                    legend_lines.append(f" {sym} - {desc}")

        gap = 2  # spaces between map and legend
        out_lines: list[str] = []
        for idx in range(max(len(map_rows), len(legend_lines))):
            left = map_rows[idx] if idx < len(map_rows) else "".ljust(self.width)
            if idx < len(legend_lines):
                out_lines.append(left + (" " * gap) + legend_lines[idx])
            else:
                out_lines.append(left)

        print("\n".join(out_lines))

    def _render_pygame(self) -> None:
        """Render to a dedicated Pygame window.

        Lazily constructs a PygameRenderer and forwards map/player/info each
        frame. If pygame isn't installed, raises a friendly RuntimeError when
        first invoked.
        """
        assert self.map is not None and self.player is not None
        # Lazy import to avoid hard dependency when not used
        if self._pg_renderer is None:
            try:
                from src.renderers import PygameRenderer
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "'pygame' render_mode requested, but pygame renderer failed to import."
                ) from e
            
            radius = Difficulty.VISION_RADII.get(self.difficulty, 3) if Difficulty else 3
            view_dim = 2 * radius + 1
            self._pg_renderer = PygameRenderer(
                view_dim,
                view_dim,
                cell_size=self._pg_cell_size,
                show_grid=self._pg_show_grid,
                show_legend=self.show_legend,
                legend_width=self._pg_legend_w,
                fps=self._pg_fps,
            )
        info = self._get_info()
        try:
            self._pg_renderer.draw(self.map, self.player, info)
        except RuntimeError as e:
            # Surface closed; drop renderer
            self._pg_renderer = None
            raise e

    def _render_rgb_array(self) -> np.ndarray:
        """Return a small RGB image of the map with the player in red.

        :return: Numpy ``uint8`` array of shape ``(H, W, 3)``.
        :rtype: numpy.ndarray
        """
        assert self.map is not None and self.player is not None
        # Simple color mapping per terrain
        colors = {
            "plains": (200, 230, 120),
            "forest": (50, 150, 50),
            "swamp": (80, 110, 80),
            "mountain": (130, 130, 130),
            "desert": (230, 210, 120),
        }
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                name = self.map.getTile(x, y).terrain.name
                img[y, x] = colors.get(name, (255, 0, 255))
        # Draw player as red
        px, py = self.player.position
        img[py, px] = (255, 50, 50)
        return img

    def close(self):
        """Close any active renderer backends/resources."""
        # Pygame renderer cleanup
        try:
            if getattr(self, "_pg_renderer", None) is not None:
                self._pg_renderer.close()
        except Exception:
            pass
        self._pg_renderer = None

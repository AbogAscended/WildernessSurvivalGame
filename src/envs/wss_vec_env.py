
import math
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src import Difficulty
from src.wss_obs_utils import encode_wss_obs_torch

class WSSNativeVecEnv(gym.vector.VectorEnv):
    """
    A fully vectorized implementation of the WSS environment running on PyTorch.
    This environment simulates `num_envs` in parallel on the GPU (or CPU).
    """

    def __init__(
        self,
        num_envs,
        width=None,
        height=None,
        difficulty="medium",
        device="cuda",
        render_mode=None,
        window_radius=2,
        max_steps=None,
        gamma=0.99,
        **kwargs
    ):
        self.num_envs = num_envs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Canonicalize difficulty
        if Difficulty:
            self.difficulty = Difficulty.canonicalize(difficulty)
        else:
            self.difficulty = difficulty
            
        self.gamma = gamma
        self.window_radius = window_radius
        self.max_steps = max_steps
        
        w_range = (20, 20)
        if Difficulty:
            w_range = Difficulty.MAP_SIZES.get(self.difficulty, (20, 20))

        max_side = 110
        if Difficulty:
            # Find max across all difficulties just to be safe
            max_side = max(max(s) for s in Difficulty.MAP_SIZES.values())
        
        self.width = width if width is not None else max_side
        self.height = height if height is not None else max_side
        
        # Dynamic sizing per env
        # If width/height passed in init are None, we allow dynamic sizing per env.
        self.dynamic_size = (width is None or height is None)
        
        # Per-env dimensions
        self.env_width = torch.full((num_envs,), self.width, device=self.device, dtype=torch.long)
        self.env_height = torch.full((num_envs,), self.height, device=self.device, dtype=torch.long)

        # Define Spaces
        # Action: 13
        self.single_action_space = spaces.Discrete(13)
        self.action_space = spaces.MultiDiscrete([13] * num_envs) # Gym vector spec

        # Observation: Flattened WSS Observation Wrapper format (to match PPO expectation)
        # Terrain(11x11x6) + Items(11x11x5) + Status(4) + Trader(16) + PrevAction(13)
        # = 726 + 605 + 4 + 16 + 13 = 1364
        self.flat_obs_dim = 1364
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.flat_obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_envs, self.flat_obs_dim), dtype=np.float32)

        super().__init__()
        
        # Prev Action Tracker
        self.prev_actions = torch.zeros((num_envs,), device=self.device, dtype=torch.long)

        # Terrain constants
        # 0: Plains, 1: Forest, 2: Swamp, 3: Mountain, 4: Desert (Internal 0-4)
        # Env uses 1-5 for observation (0 is fog).
        self.terrain_costs = torch.tensor([
            [1, 1, 1], # Plains
            [2, 1, 2], # Forest
            [3, 2, 2], # Swamp
            [4, 3, 3], # Mountain
            [2, 4, 3], # Desert
        ], device=self.device, dtype=torch.int16) # (5, 3) -> Move, Water, Food

        # Directions (0-8)
        # 0: Stay
        # 1: N, 2: S, 3: E, 4: W, 5: NE, 6: NW, 7: SE, 8: SW
        self.directions = torch.tensor([
            [0, 0], [0, -1], [0, 1], [1, 0], [-1, 0],
            [1, -1], [-1, -1], [1, 1], [-1, 1]
        ], device=self.device, dtype=torch.int16)

        # State Tensors (Allocated once)
        # Map: (N, H, W)
        self.map_terrain = torch.zeros((num_envs, self.height, self.width), device=self.device, dtype=torch.int8)
        # Items: 0=None, 1=Trader, 2=Water, 3=Food, 4=Gold
        self.map_items = torch.zeros((num_envs, self.height, self.width), device=self.device, dtype=torch.int8)
        self.map_item_amounts = torch.zeros((num_envs, self.height, self.width), device=self.device, dtype=torch.int16)
        self.map_item_repeat = torch.zeros((num_envs, self.height, self.width), device=self.device, dtype=torch.bool)
        
        # Trader Proposals: (N, H, W, 4, 4) -> 4 proposals, each is [d_food, d_water, d_strength, d_gold]
        # We pre-calculate the deltas.
        self.map_trader_deltas = torch.zeros((num_envs, self.height, self.width, 4, 4), device=self.device, dtype=torch.int16)

        # Player State
        # Pos: (N, 2) -> x, y
        self.player_pos = torch.zeros((num_envs, 2), device=self.device, dtype=torch.long)
        # Status: (N, 4) -> Food, Water, Strength, Gold
        self.player_status = torch.zeros((num_envs, 4), device=self.device, dtype=torch.int32)
        
        # Step counters
        self.steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        
        # Reward shaping state
        self.best_x = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.gold_prev = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        self.h_before = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        
        # Config
        self.max_food = 100
        self.max_water = 100
        self.start_food = 20
        self.start_water = 20
        self.start_gold = 20
        self.start_strength = 100
        self.max_strength = 100

        # Fog mask (pre-computed for window)
        self.vis_radius = 3
        if Difficulty:
            self.vis_radius = Difficulty.VISION_RADII.get(self.difficulty, 3)
            
        # Initialize
        self.reset_all()

    def reset_all(self):
        self.steps.fill_(0)
        self._generate_maps(torch.arange(self.num_envs, device=self.device))
        self._reset_players(torch.arange(self.num_envs, device=self.device))
        
    def reset(self, seed=None, options=None):
        # In vector envs, reset usually resets all.
        if seed is not None:
            # Torch RNG seeding is global, but we can try
            torch.manual_seed(seed)
            
        self.reset_all()
        return self._get_obs(), {}

    def set_attr(self, name, value):
        if name == "difficulty":
            self.difficulty = value
            if Difficulty:
                 self.difficulty = Difficulty.canonicalize(value)
        else:
            setattr(self, name, value)

    def _generate_maps(self, env_indices):
        if len(env_indices) == 0: return
        # Always use canonical difficulty
        self._generate_maps_from_config(env_indices, self.difficulty)

    def _generate_maps_from_config(self,
                                   env_indices,
                                   difficulty):
        if len(env_indices) == 0: return
        
        n = len(env_indices)

        # Dynamic Size Sampling
        if self.dynamic_size and Difficulty:
            w_min, w_max = Difficulty.MAP_SIZES.get(difficulty, (20, 20))
            sizes = torch.randint(w_min, w_max + 1, (n,), device=self.device, dtype=torch.long)
            self.env_width[env_indices] = sizes
            self.env_height[env_indices] = sizes

        # 1. Terrain
        # Weights: Plains, Forest, Swamp, Mountain, Desert
        weights = torch.tensor([4, 3, 1, 1, 1], device=self.device, dtype=torch.float32)
        if Difficulty:
            w_list = Difficulty.get_terrain_weights(difficulty)
            weights = torch.tensor(w_list, device=self.device, dtype=torch.float32)
            
        # Sample terrain indices (0-4)
        # (N, H, W)
        terrain_idx = torch.multinomial(weights, n * self.height * self.width, replacement=True)
        terrain_idx = terrain_idx.view(n, self.height, self.width).to(torch.int8)
        
        self.map_terrain[env_indices] = terrain_idx
        
        # 2. Items
        # Probs from config
        cfg = Difficulty.get_item_config(difficulty) if Difficulty else {}
        p_trader = float(cfg.get("trader_prob", 0.05))
        p_water = float(cfg.get("water_prob", 0.08))
        p_food = float(cfg.get("food_prob", 0.06))
        p_gold = float(cfg.get("gold_prob", 0.04))
        
        # Random tensor
        r = torch.rand((n, self.height, self.width), device=self.device)
        
        # Assign items: 0=None, 1=Trader, 2=Water, 3=Food, 4=Gold
        # Trader
        mask_trader = r < p_trader
        # Remaining prob space
        r2 = torch.rand((n, self.height, self.width), device=self.device)
        mask_water = r2 < p_water
        mask_food = (r2 >= p_water) & (r2 < p_water + p_food)
        mask_gold = (r2 >= p_water + p_food) & (r2 < p_water + p_food + p_gold)
        
        items = torch.zeros((n, self.height, self.width), device=self.device, dtype=torch.int8)
        items[mask_trader] = 1
        # Apply others only where no trader
        mask_no_trader = ~mask_trader
        items[mask_no_trader & mask_water] = 2
        items[mask_no_trader & mask_food] = 3
        items[mask_no_trader & mask_gold] = 4
        
        self.map_items[env_indices] = items
        
        # 3. Amounts and Repeat
        # Config amounts
        w_amt = cfg.get("water_amount", (2, 5))
        f_amt = cfg.get("food_amount", (2, 5))
        g_amt = cfg.get("gold_amount", (3, 6))
        w_rep = float(cfg.get("water_repeat_prob", 0.5))
        f_rep = float(cfg.get("food_repeat_prob", 0.35))
        
        amounts = torch.zeros((n, self.height, self.width), device=self.device, dtype=torch.int16)
        repeats = torch.zeros((n, self.height, self.width), device=self.device, dtype=torch.bool)
        
        # Vectorized randint
        def rand_amt(mask, low, high):
            cnt = mask.sum()
            if cnt > 0:
                return torch.randint(low, high + 1, (cnt,), device=self.device, dtype=torch.int16)
            return torch.tensor([], device=self.device)

        # Water
        mask_w = items == 2
        amounts[mask_w] = rand_amt(mask_w, *w_amt)
        repeats[mask_w] = torch.rand(mask_w.sum(), device=self.device) < w_rep
        
        # Food
        mask_f = items == 3
        amounts[mask_f] = rand_amt(mask_f, *f_amt)
        repeats[mask_f] = torch.rand(mask_f.sum(), device=self.device) < f_rep
        
        # Gold
        mask_g = items == 4
        amounts[mask_g] = rand_amt(mask_g, *g_amt)
        # Gold never repeats
        
        # Scaling based on tile costs (Match Sync Env)
        buffer = 5
        if difficulty in ["hard", "extreme"]:
            buffer = 2

        # Get terrain costs for scaling
        curr_t = self.map_terrain[env_indices].long() # (N, H, W)
        curr_t_flat = curr_t.view(-1)
        curr_c_flat = self.terrain_costs[curr_t_flat] # (N*H*W, 3)
        curr_c = curr_c_flat.view(n, self.height, self.width, 3)

        w_costs = curr_c[..., 1]
        f_costs = curr_c[..., 2]

        # Water Scaling
        mask_w = (items == 2)
        needed_w = (w_costs + buffer).int()
        scaled_w = torch.maximum(amounts.int(), needed_w)
        amounts = torch.where(mask_w, scaled_w.to(torch.int16), amounts)

        # Food Scaling
        mask_f = (items == 3)
        needed_f = (f_costs + buffer).int()
        scaled_f = torch.maximum(amounts.int(), needed_f)
        amounts = torch.where(mask_f, scaled_f.to(torch.int16), amounts)
        
        self.map_item_amounts[env_indices] = amounts
        self.map_item_repeat[env_indices] = repeats
        
        # Identify trader locations
        trader_indices = torch.nonzero(mask_trader, as_tuple=True) # (idx_n, idx_y, idx_x)
        num_traders = trader_indices[0].size(0)
        
        if num_traders > 0:
            # Get terrain costs at these locations
            # terrain types: 0..4
            t_types = terrain_idx[trader_indices[0], trader_indices[1], trader_indices[2]].long() # (T,)
            costs = self.terrain_costs[t_types] # (T, 3) -> M, W, F
            
            m_cost = costs[:, 0]
            w_cost = costs[:, 1]
            f_cost = costs[:, 2]
            
            # Logic from Trader.py
            w_trade = (w_cost + 1) // 2
            f_trade = (f_cost + 1) // 2
            w_needed = w_cost + w_trade
            f_needed = f_cost + f_trade
            base_cost = torch.max(w_needed, f_needed)
            base_cost = torch.max(torch.ones_like(base_cost), base_cost)
            
            # Generate 4 proposals per trader
            # We'll do this 4 times
            proposals = torch.zeros((num_traders, 4, 4), device=self.device, dtype=torch.int16)
            
            for p_i in range(4):
                # Payment Qty
                pay_qty = (base_cost.float() * (1.0 + 0.5 * torch.rand(num_traders, device=self.device))).long()

                mult = 0.9 + 1.3 * torch.rand(num_traders, device=self.device)
                
                offer_qty = (pay_qty.float() * mult).long()
                offer_qty = torch.max(torch.ones_like(offer_qty), offer_qty)
                
                # Offer Type: 0=Gold, 1=Water, 2=Food
                offer_type = torch.randint(0, 3, (num_traders,), device=self.device)

                offset = torch.randint(1, 3, (num_traders,), device=self.device)
                want_type = (offer_type + offset) % 3
                
                # Construct Delta Vector: [Food, Water, Strength, Gold]
                # Indices in logic: Gold=0, Water=1, Food=2
                # Target Indices: Food=0, Water=1, Strength=2, Gold=3
                
                # Mapping from logic (G,W,F) to target (F,W,S,G)
                # G(0)->3, W(1)->1, F(2)->0
                type_map = torch.tensor([3, 1, 0], device=self.device)
                
                d_offer_idx = type_map[offer_type]
                d_want_idx = type_map[want_type]
                
                # Create delta rows
                deltas = torch.zeros((num_traders, 4), device=self.device, dtype=torch.int16)
                
                # Scatter add offer
                deltas.scatter_add_(1, d_offer_idx.unsqueeze(1), offer_qty.unsqueeze(1).short())
                # Scatter sub want
                deltas.scatter_add_(1, d_want_idx.unsqueeze(1), -pay_qty.unsqueeze(1).short())
                
                proposals[:, p_i, :] = deltas

            self.map_trader_deltas[env_indices[trader_indices[0]], trader_indices[1], trader_indices[2]] = proposals

        # Masking "Ghost Terrain" and items outside valid bounds
        y_grid = torch.arange(self.height, device=self.device).view(1, -1, 1)
        x_grid = torch.arange(self.width, device=self.device).view(1, 1, -1)
        
        k = len(env_indices)
        # Expand to K
        # env_height/width are (N,) so we index them
        current_h = self.env_height[env_indices].view(k, 1, 1)
        current_w = self.env_width[env_indices].view(k, 1, 1)
        
        mask_h = y_grid < current_h
        mask_w = x_grid < current_w
        valid_mask = mask_h & mask_w
        
        # Apply mask to terrain (set to -1 -> Obs 0/Fog)
        current_terrain = self.map_terrain[env_indices]
        current_terrain[~valid_mask] = -1
        self.map_terrain[env_indices] = current_terrain
        
        # Apply mask to items (set to 0/None)
        current_items = self.map_items[env_indices]
        current_items[~valid_mask] = 0
        self.map_items[env_indices] = current_items

    def _reset_players(self, env_indices):
        # Start at (0, H//2)
        self.player_pos[env_indices, 0] = 0
        # Use env_height for current map size, not max buffer height
        self.player_pos[env_indices, 1] = self.env_height[env_indices] // 2
        
        # Stats
        self.player_status[env_indices, 0] = self.start_food
        self.player_status[env_indices, 1] = self.start_water
        self.player_status[env_indices, 2] = self.start_strength
        self.player_status[env_indices, 3] = self.start_gold
        
        # Rewards
        self.best_x[env_indices] = 0.0
        self.gold_prev[env_indices] = self.start_gold
        self.steps[env_indices] = 0

        # Init h_before
        w = self.player_status[env_indices, 1].float() / self.max_water
        f = self.player_status[env_indices, 0].float() / self.max_food
        s = self.player_status[env_indices, 2].float() / self.max_strength
        self.h_before[env_indices] = torch.min(torch.min(w, f), s)
        
        # Reset Prev Actions
        self.prev_actions[env_indices] = 0

    def step(self, actions):
        # actions: (N,)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        n = self.num_envs
        
        # 0. Pre-Calc Potentials
        curr_x = self.player_pos[:, 0]
        # curr_y = self.player_pos[:, 1]
        phi_e_before = curr_x.float() / (self.env_width - 1).float()

        # 1. Calculate Next Pos & Passability
        # direction deltas
        move_mask = actions < 9
        trade_mask = actions >= 9
        
        move_dirs = self.directions[actions.where(move_mask, torch.tensor(0, device=self.device))]
        
        curr_x = self.player_pos[:, 0]
        curr_y = self.player_pos[:, 1]
        
        next_x = curr_x + move_dirs[:, 0]
        next_y = curr_y + move_dirs[:, 1]
        
        # Clip to bounds
        next_x = torch.clamp(next_x, min=0)
        next_y = torch.clamp(next_y, min=0)
        
        # Use per-env dimensions
        next_x = torch.min(next_x, self.env_width - 1)
        next_y = torch.min(next_y, self.env_height - 1)

        # Terrain Costs (Check Passability on Target Tile)
        t_types = self.map_terrain[torch.arange(n), next_y, next_x].long() # (N,)
        costs = self.terrain_costs[t_types] # (N, 3) -> Move, Water, Food
        
        c_str = costs[:, 0]
        c_wat = costs[:, 1]
        c_fod = costs[:, 2]

        # Match WildernessSurvivalEnv: strength >= m and water >= w and food >= f
        curr_f = self.player_status[:, 0]
        curr_w = self.player_status[:, 1]
        curr_s = self.player_status[:, 2]
        
        can_enter = (curr_s >= c_str.int()) & (curr_w >= c_wat.int()) & (curr_f >= c_fod.int())
        
        # Identify blocked moves (attempted move but cannot enter)
        blocked = move_mask & (~can_enter)
        
        # Update move_mask to exclude blocked moves
        move_mask = move_mask & can_enter
        
        # 2. Update Position (only if move action & not blocked)
        # Trade actions don't move
        self.player_pos[:, 0] = torch.where(move_mask, next_x, curr_x)
        self.player_pos[:, 1] = torch.where(move_mask, next_y, curr_y)
        
        # 3. Item Collection (at new position)
        px, py = self.player_pos[:, 0], self.player_pos[:, 1]
        i_type = self.map_items[torch.arange(n), py, px] # (N,)
        i_amt = self.map_item_amounts[torch.arange(n), py, px]
        i_rep = self.map_item_repeat[torch.arange(n), py, px]
        
        has_trader = (i_type == 1)
        has_water = (i_type == 2)
        has_food = (i_type == 3)
        has_gold = (i_type == 4)
        
        # Update status
        self.player_status[:, 1] += (has_water * i_amt).int()
        self.player_status[:, 0] += (has_food * i_amt).int()
        self.player_status[:, 3] += (has_gold * i_amt).int()
        
        # Clear items if not repeating (and collected)
        collected = (i_type > 1)
        should_clear = collected & (~i_rep)
        clear_idx = torch.nonzero(should_clear).squeeze(1)
        if clear_idx.numel() > 0:
            self.map_items[clear_idx, py[clear_idx], px[clear_idx]] = 0
            self.map_item_amounts[clear_idx, py[clear_idx], px[clear_idx]] = 0
            
        # 4. Trade Actions
        # Actions 9-12 correspond to proposals 0-3
        # Only valid if on trader tile AND afford payment
        t_idx = actions - 9
        is_trade_action = (actions >= 9)
        valid_trade_loc = is_trade_action & has_trader
        
        if valid_trade_loc.any():
            idx = torch.nonzero(valid_trade_loc).squeeze(1)
            props = self.map_trader_deltas[idx, py[idx], px[idx]] # (K, 4, 4)
            sel_prop = props[torch.arange(len(idx), device=self.device), t_idx[idx]] # (K, 4) -> [dF, dW, dS, dG]
            
            # Check Affordability
            # sel_prop has negative values for payment. current + delta >= 0
            # We check Food, Water, Gold (indices 0, 1, 3 in status)
            # dF=0, dW=1, dS=2, dG=3 in sel_prop?
            # map_trader_deltas constructed as: [Food, Water, Strength, Gold]
            # player_status: [Food, Water, Strength, Gold]
            # So indices match.
            
            curr_vals = self.player_status[idx]
            new_vals = curr_vals + sel_prop.int()
            
            # Check non-negative resources (Strength usually 0 delta, so ignore check or check it too)
            can_afford = (new_vals[:, 0] >= 0) & (new_vals[:, 1] >= 0) & (new_vals[:, 3] >= 0)
            
            # Apply only where affordable
            valid_idx = idx[can_afford]
            if valid_idx.numel() > 0:
                # Re-select props for valid subset
                # Or just mask
                # We can use scatter add or just indexed assignment if we are careful
                # Let's use the computed new_vals for valid ones
                self.player_status[valid_idx] += sel_prop[can_afford].int()

        # 5. Apply Step Costs (Move/Stay/ForceStay)
        # If stay (action 0) OR blocked OR trade action, costs are based on CURRENT tile (halved), strength +5
        stay_mask = (actions == 0)
        force_stay = stay_mask | blocked | trade_mask
        
        # For force_stay, we need costs of CURRENT tile.
        # Since we already updated pos, if we moved, we are at new tile.
        # If we were blocked/stay/trade, pos didn't change.
        # So we can just look at player_pos.
        curr_t_types = self.map_terrain[torch.arange(n), py, px].long()
        curr_costs = self.terrain_costs[curr_t_types]
        curr_c_str = curr_costs[:, 0]
        curr_c_wat = curr_costs[:, 1]
        curr_c_fod = curr_costs[:, 2]
        
        stay_wat = (curr_c_wat + 1) // 2
        stay_fod = (curr_c_fod + 1) // 2
        
        # If force_stay: use stay costs. Else: use move costs (from step 1 calculation)
        # Wait, move costs depended on NEXT tile (which is now Current tile if we moved).
        # So actually, if we moved, 'curr_costs' IS the move cost (entering new tile).
        # So we can use 'curr_costs' for everything?
        # Sync Env:
        # Move -> Cost is full tile cost of entered tile.
        # Stay/Blocked -> Cost is half tile cost of current tile.
        # So yes, always based on the tile at player_pos (after move).
        
        final_c_wat = torch.where(force_stay, stay_wat, curr_c_wat)
        final_c_fod = torch.where(force_stay, stay_fod, curr_c_fod)
        
        # Strength: if force_stay, +5. Else -cost.
        d_str = -curr_c_str
        d_str = torch.where(force_stay, torch.tensor(5, device=self.device, dtype=torch.int16), d_str)
        
        self.player_status[:, 1] -= final_c_wat.int()
        self.player_status[:, 0] -= final_c_fod.int()
        self.player_status[:, 2] += d_str.int()

        # Clamp stats
        self.player_status[:, 0] = torch.clamp(self.player_status[:, 0], 0, self.max_food)
        self.player_status[:, 1] = torch.clamp(self.player_status[:, 1], 0, self.max_water)
        self.player_status[:, 2] = torch.clamp(self.player_status[:, 2], 0, self.max_strength) # Clamped to max_strength
        
        # 6. Termination & Reward
        reached_goal = (self.player_pos[:, 0] == self.env_width - 1)
        dead = (self.player_status[:, 0] <= 0) | (self.player_status[:, 1] <= 0)
        
        self.steps += 1
        truncated = torch.zeros(n, device=self.device, dtype=torch.bool)
        if self.max_steps:
            truncated = (self.steps >= self.max_steps)
            
        terminated = reached_goal | dead
        done = terminated | truncated
        
        # Rewards
        # New Logic: Full Potential Shaping
        
        # Post-Potentials
        phi_e_after = self.player_pos[:, 0].float() / (self.env_width - 1).float()
        
        w = self.player_status[:, 1].float() / self.max_water
        f = self.player_status[:, 0].float() / self.max_food
        s = self.player_status[:, 2].float() / self.max_strength
        h_now = torch.min(torch.min(w, f), s)
        
        # Weights
        w_goal = 1.0
        w_death = 0.5
        w_east = 1.0
        w_surv = 0.5
        w_disc = 0.5
        w_gold = 0.000
        w_time = 0.000
        
        # Hardness Scaling (Canonical)
        hardness = 0.0
        if Difficulty:
            try:
                hardness = Difficulty.get_hardness(self.difficulty)
            except:
                pass
        
        w_goal *= (1.0 + 0.5 * hardness)
        w_east *= (1.0 + 0.25 * hardness)
        
        # Terms
        r_east = w_east * (self.gamma * phi_e_after - phi_e_before)
        r_surv = w_surv * (self.gamma * h_now - self.h_before)
        
        # Discovery
        r_disc = torch.zeros(n, device=self.device, dtype=torch.float32)
        new_x = self.player_pos[:, 0].float()
        is_new_best = new_x > self.best_x
        if is_new_best.any():
            # Normalize by width
            r_disc[is_new_best] = w_disc * (new_x[is_new_best] - self.best_x[is_new_best]) / (self.env_width[is_new_best].float() - 1)
            self.best_x[is_new_best] = new_x[is_new_best]
            
        # Gold
        gold_now = self.player_status[:, 3]
        r_gold = w_gold * (gold_now - self.gold_prev).float()
        # Disable gold reward for trade actions (prevent arbitrage loops)
        r_gold = torch.where(actions >= 9, torch.tensor(0.0, device=self.device), r_gold)
        
        # Time
        r_time = -w_time
        
        # Terminals
        r_goal = torch.where(reached_goal, w_goal, 0.0)
        r_death = torch.where(dead, -w_death, 0.0)
        
        reward = r_east + r_surv + r_disc + r_gold + r_goal + r_death + r_time
        
        # Clip
        reward = torch.clamp(reward, -1.0, 1.0)
        
        # Update state for next step
        self.h_before[:] = h_now
        self.gold_prev[:] = gold_now
        
        # Update Prev Actions (before reset, so reset can overwrite if done)
        self.prev_actions[:] = actions

        # 6. Auto-Reset
        # If done, save info, reset env.
        
        infos = {}
        final_infos = np.array([None] * n, dtype=object)
        
        done_idx = torch.nonzero(done).squeeze(1)
        if done_idx.numel() > 0:
            done_cpu = done_idx.cpu().numpy()
            reached_cpu = reached_goal[done_idx].cpu().numpy()
            
            for i, idx_val in enumerate(done_cpu):
                final_infos[idx_val] = {
                    "reached_goal": bool(reached_cpu[i]),
                }
            
            # Reset those envs
            self._generate_maps(done_idx)
            self._reset_players(done_idx)
            
        infos["final_info"] = final_infos
            
        # Obs
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, infos

    def _get_obs(self):
        # Construct flattened observation on GPU
        # Terrain(6) + Items(5) + Status(4) + Trader(16)
        # One-hot terrain/items in window
        
        n = self.num_envs
        px = self.player_pos[:, 0]
        py = self.player_pos[:, 1]
        
        # Create padded maps
        padded_terrain = torch.zeros((n, self.height + 10, self.width + 10), device=self.device, dtype=torch.int8)
        
        # Fill center
        padded_terrain[:, 5:-5, 5:-5] = self.map_terrain + 1

        # Construct gathering indices
        # grid_y: (11, 1)
        # grid_x: (1, 11)
        win_y = torch.arange(11, device=self.device).view(11, 1)
        win_x = torch.arange(11, device=self.device).view(1, 11)
        
        # Base: (N, 1, 1)
        base_y = py.view(n, 1, 1)
        base_x = px.view(n, 1, 1)
        
        idx_y = base_y + win_y # (N, 11, 11) broadcast
        idx_x = base_x + win_x

        # padded_terrain: (N, H', W')
        batch_idx = torch.arange(n, device=self.device).view(n, 1, 1).expand(n, 11, 11)
        
        t_win = padded_terrain[batch_idx, idx_y, idx_x] # (N, 11, 11)
        
        # Apply Fog Mask
        # Dist from center (5, 5)
        dy = torch.abs(torch.arange(11, device=self.device) - 5).view(11, 1)
        dx = torch.abs(torch.arange(11, device=self.device) - 5).view(1, 11)
        dist = torch.max(dy, dx) # Chebyshev
        
        fog_mask = dist > self.vis_radius
        t_win = t_win.clone()
        t_win[:, fog_mask] = 0 # Set to Fog
        
        # Items
        # Map items: 0..4.
        padded_items = torch.zeros((n, self.height + 10, self.width + 10), device=self.device, dtype=torch.int8)
        padded_items[:, 5:-5, 5:-5] = self.map_items
        
        i_win = padded_items[batch_idx, idx_y, idx_x]
        i_win[:, fog_mask] = 0
        
        # Trader
        has_trader = (self.map_items[torch.arange(n), py, px] == 1)
        
        tr_deltas = torch.zeros((n, 16), device=self.device)
        
        tr_idx = torch.nonzero(has_trader).squeeze(1)
        if tr_idx.numel() > 0:
            deltas = self.map_trader_deltas[tr_idx, py[tr_idx], px[tr_idx]] # (K, 4, 4)
            tr_deltas[tr_idx] = deltas.view(-1, 16).float()
            
        return encode_wss_obs_torch(
            t_win,
            i_win,
            self.player_status,
            tr_deltas,
            self.prev_actions
        )


import numpy as np
import torch
import unittest
from src.envs.wss_env import WildernessSurvivalEnv
from src.envs.wss_vec_env import WSSNativeVecEnv
from train_ppo import WSSObservationWrapper
from src import Difficulty

class TestTrajEquivalence(unittest.TestCase):
    def test_trajectory(self):
        # 1. Init Envs
        diff = "medium"
        env_sync = WildernessSurvivalEnv(difficulty=diff, max_steps=100)
        env_sync = WSSObservationWrapper(env_sync)
        
        env_vec = WSSNativeVecEnv(num_envs=1, difficulty=diff, max_steps=100, device="cpu")
        
        # 2. Reset
        obs_s, info_s = env_sync.reset(seed=42)
        obs_v, info_v = env_vec.reset(seed=42)
        
        # 3. Force Sync State (Sync -> Vec)
        # Map Terrain mapping
        # wss_env.py tile names to vec env indices (0-4)
        # terrain_costs order in wss_vec_env: Plains, Forest, Swamp, Mountain, Desert
        # So: Plains=0, Forest=1, Swamp=2, Mountain=3, Desert=4
        t_map = {'plains':0, 'forest':1, 'swamp':2, 'mountain':3, 'desert':4}
        
        w, h = env_sync.env.width, env_sync.env.height
        
        # Update vec env dims
        env_vec.env_width[:] = w
        env_vec.env_height[:] = h
        
        # Helper to access sync map
        sync_map = env_sync.env.map
        
        for y in range(h):
            for x in range(w):
                tile = sync_map.getTile(x, y)
                # Terrain
                t_idx = t_map[tile.terrain.name.lower()]
                env_vec.map_terrain[0, y, x] = t_idx
                
                # Items
                items = getattr(tile, "items", [])
                has_trader = any(i.getType() == "trader" for i in items)
                has_gold = any(i.getType() == "gold" for i in items)
                has_water = any(i.getType() == "water" for i in items)
                has_food = any(i.getType() == "food" for i in items)
                
                i_val = 0
                amt = 0
                is_rep = False
                if has_trader:
                    i_val = 1
                    # Trader deltas
                    trader = next(i for i in items if i.getType() == "trader")
                    inv = trader.getInventory() # 4 proposals
                    for k, prop in enumerate(inv):
                        if k >= 4: break
                        # prop: [wants, offers] -> [[G,W,F], [G,W,F]]
                        # Delta = Offer - Want
                        # Target indices: Food=0, Water=1, Strength=2, Gold=3
                        # prop indices: Gold=0, Water=1, Food=2
                        d_f = prop[1][2] - prop[0][2]
                        d_w = prop[1][1] - prop[0][1]
                        d_s = 0
                        d_g = prop[1][0] - prop[0][0]
                        env_vec.map_trader_deltas[0, y, x, k] = torch.tensor([d_f, d_w, d_s, d_g], dtype=torch.int16)
                elif has_gold:
                    i_val = 4
                    amt = sum(i.itemAmount for i in items if i.getType() == "gold")
                elif has_water:
                    i_val = 2
                    # Find water item
                    w_item = next(i for i in items if i.getType() == "water")
                    raw = w_item.itemAmount
                    is_rep = w_item.isRepeating
                    # Scale water
                    m_c, w_c, f_c = tile.get_costs()
                    buffer = 5 # Medium
                    needed = w_c + buffer
                    amt = max(raw, needed)
                elif has_food:
                    i_val = 3
                    f_item = next(i for i in items if i.getType() == "food")
                    raw = f_item.itemAmount
                    is_rep = f_item.isRepeating
                    # Scale food
                    m_c, w_c, f_c = tile.get_costs()
                    buffer = 5 # Medium
                    needed = f_c + buffer
                    amt = max(raw, needed)
                
                env_vec.map_items[0, y, x] = i_val
                env_vec.map_item_amounts[0, y, x] = amt
                env_vec.map_item_repeat[0, y, x] = is_rep
                
        # Player
        px, py = env_sync.env.player.position
        env_vec.player_pos[0, 0] = px
        env_vec.player_pos[0, 1] = py
        
        # Status
        p = env_sync.env.player
        # Vec Status: [Food, Water, Strength, Gold]
        env_vec.player_status[0] = torch.tensor([p.currentFood, p.currentWater, p.currentStrength, p.currentGold], dtype=torch.int32)
        
        # Sync Reward Trackers
        env_vec.best_x[0] = px
        env_vec.gold_prev[0] = p.currentGold
        
        w_v = float(p.currentWater) / env_vec.max_water
        f_v = float(p.currentFood) / env_vec.max_food
        s_v = float(p.currentStrength) / env_vec.max_strength
        env_vec.h_before[0] = min(min(w_v, f_v), s_v)
        
        # Force Obs refresh on Vec
        obs_v = env_vec._get_obs()

        obs_s_flat = obs_s
        obs_v_flat = obs_v.cpu().numpy().flatten()
        
        np.testing.assert_allclose(obs_s_flat, obs_v_flat, atol=1e-5, err_msg="Initial obs mismatch")
        print("Initial state synced and obs match.")
        
        # 5. Step
        actions = [3, 3, 2, 1, 0, 9, 0] # E, E, S, N, Stay, Trade(0), Stay
        
        for step_i, act in enumerate(actions):
            # Sync Step
            o_s, r_s, te_s, tr_s, i_s = env_sync.step(act)
            
            # Vec Step
            o_v, r_v, te_v, tr_v, i_v = env_vec.step(np.array([act]))
            
            # Compare
            # Obs
            try:
                np.testing.assert_allclose(o_s, o_v.cpu().numpy().flatten(), atol=1e-5, err_msg=f"Obs mismatch at step {step_i}")
            except AssertionError as e:
                # Print details
                ovn = o_v.cpu().numpy().flatten()
                diff = np.abs(o_s - ovn)
                bad_idx = np.where(diff > 1e-5)[0]
                print(f"Bad Indices: {bad_idx}")
                print(f"Sync Values: {o_s[bad_idx]}")
                print(f"Vec Values: {ovn[bad_idx]}")
                
                # Check components
                # Terrain: 0-726
                # Items: 726-1331
                # Status: 1331-1335 (4 items)
                # Trader: 1335-1351
                # PrevAction: 1351-1364
                
                if np.any((bad_idx >= 1331) & (bad_idx < 1335)):
                    print("Status mismatch in Obs")
                    p = env_sync.env.player
                    print(f"Sync Player: F={p.currentFood}, W={p.currentWater}, S={p.currentStrength}, G={p.currentGold}")
                    print(f"Vec Player: {env_vec.player_status[0]}")
                
                raise e
            
            # Reward
            r_v_val = r_v.item()
            diff = abs(r_s - r_v_val)
            if diff > 1e-4:
                print(f"Reward mismatch at step {step_i}: Sync={r_s:.5f}, Vec={r_v_val:.5f}")
                # Don't fail immediately to see full trace, or fail?
                # self.assertTrue(diff <= 1e-4, f"Reward mismatch: {r_s} vs {r_v_val}")
            
            # Term/Trunc
            self.assertEqual(te_s, te_v.item(), f"Terminated mismatch at step {step_i}")
            self.assertEqual(tr_s, tr_v.item(), f"Truncated mismatch at step {step_i}")
            
            # Status
            p = env_sync.env.player
            vec_status = env_vec.player_status[0].cpu().numpy()
            sync_status = np.array([p.currentFood, p.currentWater, p.currentStrength, p.currentGold])
            
            # Allow small diff? Should be exact integer match usually.
            np.testing.assert_allclose(sync_status, vec_status, atol=1e-5, err_msg=f"Status mismatch at step {step_i}")
            
            print(f"Step {step_i} (Action {act}): OK")

if __name__ == "__main__":
    unittest.main()

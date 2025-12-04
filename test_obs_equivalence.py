import numpy as np
import torch
import unittest
from src.wss_obs_utils import encode_wss_obs_numpy, encode_wss_obs_torch

class TestObsEquivalence(unittest.TestCase):
    def test_encoding_parity(self):
        # 1. Create synthetic data
        # Terrain: 11x11 int 0..5
        t_np = np.random.randint(0, 6, (11, 11), dtype=np.int32)
        # Items: 11x11 int 0..4
        i_np = np.random.randint(0, 5, (11, 11), dtype=np.int32)
        # Status: [Food, Water, Strength, Gold]
        s_np = np.array([50, 60, 80, 100], dtype=np.int32)
        # Trader: 4x4, random values
        tr_np = np.random.randint(-10, 10, (4, 4), dtype=np.int32)
        # Prev Action: 0..12
        pa_int = 5
        
        # 2. Encode with Numpy
        obs_np = encode_wss_obs_numpy(t_np, i_np, s_np, tr_np, pa_int)
        
        # 3. Prepare Torch inputs (Batch size 1)
        t_th = torch.tensor(t_np, dtype=torch.int16).unsqueeze(0) # (1, 11, 11)
        i_th = torch.tensor(i_np, dtype=torch.int16).unsqueeze(0) # (1, 11, 11)
        s_th = torch.tensor(s_np, dtype=torch.int32).unsqueeze(0) # (1, 4)
        
        # Trader deltas expected as (N, 16) or (N, 4, 4)
        tr_th = torch.tensor(tr_np, dtype=torch.int32).unsqueeze(0) # (1, 4, 4)
        # encode_wss_obs_torch handles view(-1)
        
        pa_th = torch.tensor([pa_int], dtype=torch.long) # (1,)
        
        # 4. Encode with Torch
        obs_th = encode_wss_obs_torch(t_th, i_th, s_th, tr_th, pa_th)
        
        # 5. Compare
        obs_th_np = obs_th.cpu().numpy().flatten()
        
        print(f"Numpy shape: {obs_np.shape}")
        print(f"Torch shape: {obs_th_np.shape}")
        
        np.testing.assert_allclose(obs_np, obs_th_np, rtol=1e-5, atol=1e-5)
        print("Observation encoding match confirmed!")

if __name__ == "__main__":
    unittest.main()

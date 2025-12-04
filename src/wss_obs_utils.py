import numpy as np
import torch

OBS_TERRAIN_SIZE = 726
OBS_ITEMS_SIZE = 605
OBS_STATUS_SIZE = 4
OBS_TRADER_SIZE = 16
OBS_PREV_ACTION_SIZE = 13
OBS_TOTAL_SIZE = 1364

def encode_wss_obs_numpy(terrain_11x11: np.ndarray, 
                         items_11x11: np.ndarray, 
                         status_4: np.ndarray, 
                         trader_4x4: np.ndarray, 
                         prev_action_int: int) -> np.ndarray:
    """
    Encodes the WSS observation components into a flattened 1364-d vector (NumPy).
    
    Args:
        terrain_11x11: (11, 11) int array, values 0..5
        items_11x11: (11, 11) int array, values 0..4
        status_4: (4,) int array/list: [Food, Water, Strength, Gold]
        trader_4x4: (4, 4) int array/list: 4 proposals x [dFood, dWater, dStrength, dGold]
        prev_action_int: scalar int (0..12)
        
    Returns:
        (1364,) float32 array
    """
    # 1. Terrain One-Hot: 6 categories (0..5) -> 726 floats
    t = terrain_11x11.flatten()
    t_oh = np.eye(6)[t].flatten()

    # 2. Items One-Hot: 5 categories (0..4) -> 605 floats
    i = items_11x11.flatten()
    i_oh = np.eye(5)[i].flatten()

    # 3. Status: 4 floats / 100.0
    s = np.array(status_4, dtype=np.float32) / 100.0

    # 4. Trader: 16 floats * 0.1
    tr = np.array(trader_4x4, dtype=np.float32).flatten() * 0.1

    # 5. Prev Action: One-Hot 13 categories
    pa_oh = np.eye(13)[prev_action_int].flatten()

    return np.concatenate([t_oh, i_oh, s, tr, pa_oh]).astype(np.float32)

def encode_wss_obs_torch(t_win: torch.Tensor, 
                         i_win: torch.Tensor, 
                         status: torch.Tensor, 
                         trader_deltas: torch.Tensor, 
                         prev_actions: torch.Tensor) -> torch.Tensor:
    """
    Encodes the WSS observation components into a flattened tensor (PyTorch).
    
    Args:
        t_win: (N, 11, 11) int tensor, values 0..5
        i_win: (N, 11, 11) int tensor, values 0..4
        status: (N, 4) int/float tensor
        trader_deltas: (N, 16) or (N, 4, 4) float tensor (already extracted for current tile)
        prev_actions: (N,) int tensor (0..12)
        
    Returns:
        (N, 1364) float tensor
    """
    n = t_win.shape[0]
    
    # 1. Terrain One-Hot
    t_oh = torch.nn.functional.one_hot(t_win.long(), num_classes=6) # (N, 11, 11, 6)
    t_flat = t_oh.view(n, -1).float()
    
    # 2. Items One-Hot
    i_oh = torch.nn.functional.one_hot(i_win.long(), num_classes=5) # (N, 11, 11, 5)
    i_flat = i_oh.view(n, -1).float()
    
    # 3. Status
    s_flat = status.float() / 100.0
    
    # 4. Trader
    # Input expected to be (N, 16) or (N, 4, 4)
    tr_flat = trader_deltas.view(n, -1).float() * 0.1
    
    # 5. Prev Action One-Hot
    pa_oh = torch.nn.functional.one_hot(prev_actions.long(), num_classes=13).float()
    
    return torch.cat([t_flat, i_flat, s_flat, tr_flat, pa_oh], dim=1)

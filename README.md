# WildernessSurvivalGame

Implementation of the Wilderness Survival System (WSS) for Reinforcement Learning.

This repo provides:
- A playable core of the WSS (map, tiles, terrain, items, trader, player).
- Multiple renderers you can toggle: ASCII (NetHack-style), RGB array, and a windowed Pygame renderer.
- A Gymnasium-compatible environment wrapper (`WildernessSurvivalEnv`).
- A TorchRL-ready PPO training scaffold with a simple, well‑commented policy network.
- A comprehensive CLI for training, curriculum learning, and watching agents.

**Reward design**: The environment implements a shaped reward aligned with “reach the east edge as fast as possible.” It combines potential-based eastward progress, survival improvement (via water/food/strength health), a small discovery bonus for surpassing the best x so far, a terminal goal bonus (+1) or death penalty, and a tiny bonus for increases in gold. Rewards are mildly scaled by difficulty hardness and clipped to [-1, 1] for PPO stability.

## Install

1. Python 3.10+ is recommended.
2. From the project root, install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

**Packages used**:
- `gymnasium` for the modern RL environment API
- `numpy` for vector ops
- `torch` and `torchrl` for PPO and model definition
- `pygame` (optional) for the new windowed renderer; if not installed, use ASCII or RGB render modes

## Quick Start (Manual Run)

Run a quick interactive episode with optional rendering to verify the environment works:

```bash
python run_env.py
```

You will be prompted for:
- Map width/height
- Difficulty: `easy`, `medium`, `hard`, `extreme`
- Render mode: `none` (fast), `human` (ASCII), `rgb_array`, or `pygame` (windowed)

## Human Play

You can play the game yourself using the terminal or a Pygame window.

```bash
python play_human.py --difficulty medium --backend pygame
```

**Arguments**:
- `--backend`: `ascii` (terminal) or `pygame` (windowed).
- `--width`, `--height`: Override default map sizes.
- `--difficulty`: Sets terrain/item probabilities.
- `--colors`, `--clear-screen`: For ASCII mode.

**Controls**:
- **Movement**: `WASD` or `h/j/k/l` (cardinal), `q/e/z/c` or `y/u/b/n` (diagonals).
- **Wait/Rest**: `.` or `Space`. Regains strength but consumes food/water.
- **Trade**: `1`, `2`, `3`, `4` to accept trader proposals when adjacent to a trader.
- **Quit**: `Q` or `Esc`.

## Environment Details

```python
from src.envs.wss_env import WildernessSurvivalEnv

env = WildernessSurvivalEnv(difficulty="medium", render_mode="human")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

### Observation Space (Dict)

The observation is a dictionary containing local views and player status:
- `terrain`: 11x11 grid of terrain types (0=Fog, 1=Plains, 2=Forest, 3=Swamp, 4=Mountain, 5=Desert).
- `items`: 11x11 grid of items (0=None, 1=Trader, 2=Water, 3=Food, 4=Gold).
- `status`: 4-vector `[Food, Water, Strength, Gold]`.
- `trader`: 4x4 matrix representing up to 4 trade offers (deltas for Food, Water, Strength, Gold).
- `prev_action`: Scalar indicating the last action taken.

### Action Space (Discrete 13)

- **0**: Stay/Rest
- **1-8**: Movement (N, S, E, W, NE, NW, SE, SW)
- **9-12**: Trade Options 1-4 (corresponds to the offers in `obs['trader']`)

### Termination & Truncation
- **Success**: Reach east edge (x == width-1).
- **Death**: Strength, Water, or Food drops to <= 0.
- **Truncation**: Optional max steps cap (default varies).

### Reward Design
The reward is wired into `WildernessSurvivalEnv.step()`:
- **Eastward progress**: Potential-based; pays for net progress east.
- **Survival**: Potential-based; rewards increasing `min(water%, food%, strength%)`.
- **Discovery**: Bonus for reaching a new furthest 'x' coordinate.
- **Gold**: Tiny bonus for collecting gold.
- **Goal**: +1.0 (scaled by difficulty).
- **Death**: Penalty (scaled by difficulty).
- **Time**: Small negative step penalty to encourage speed.

## Training (PPO)

### 1. Basic Training
Run the simple baseline script:

```bash
python train_ppo.py
```
This runs a PPO loop using a commented-out policy network in `src/Brain.py` (or defined inline).

### 2. Advanced CLI Training (Recommended)
Use `train_cli.py` for a fully configurable run with Curriculum Learning, WandB logging, and Checkpointing.

```bash
python train_cli.py \
  --total-steps 1000000 \
  --num-envs 16 \
  --difficulty easy \
  --wandb-project MyWSSProject \
  --checkpoint-freq 500000
```

**Key Arguments**:
- `--num-envs`: Number of parallel environments (vectorized).
- `--vector-mode`: `sync` (serial) or `native` (torchrl specialized).
- `--difficulty`: Starting difficulty.
- `--success-thresh-up`: Win rate required to increase difficulty curriculum.
- `--render-mode`: `human` to watch training (slow!), `none` for speed.

### 3. Watch a Trained Agent
Visualize a trained model playing the game:

```bash
python watch_agent.py --checkpoint checkpoints/model_final.pt --difficulty hard --render-mode pygame
```

## Difficulty System

The environment uses a centralized difficulty spec in `src/Difficulty.py`.

**Levels**: `easy`, `medium`, `hard`, `extreme`.

**Scales with difficulty**:
- **Map Size**: Larger maps for harder levels.
- **Vision Radius**: Reduced visibility in harder levels (e.g., Easy=4, Extreme=1).
- **Terrain**: More mountains/swamps, fewer plains.
- **Items**: Scarcity of food/water, abundance of gold changes.
- **Trader**: Less lenient deals, fewer items offered.

## Renderers

### ASCII (`render_mode="human"`)
NetHack-style view.
- Symbols: `@` Player, `R` Trader, `$` Gold, `w` Water, `%` Food.
- Terrain: `.` Plains, `T` Forest, `~` Swamp, `^` Mountain, `:` Desert.

### Pygame (`render_mode="pygame"`)
Dedicated window.
- Graphical tiles with grid.
- HUD with resource bars.
- Visual overlays for trade offers.
- Resizable window and fullscreen support (F11).

## File Map

- `src/Difficulty.py` — Centralized difficulty configurations (map size, vision, spawn rates).
- `src/Terrain.py` — Terrain definitions.
- `src/Tiles.py` — Map tile logic (costs, item collection).
- `src/Items.py` — Item definitions.
- `src/Trader.py` — Trader logic and inventory generation.
- `src/Player.py` — Player state (resources, position).
- `src/Map.py` — Map generation (cellular automata/noise).
- `src/envs/wss_env.py` — Main Gymnasium environment wrapper.
- `src/renderers/pygame_renderer.py` — Pygame backend.
- `play_human.py` — Human play entry point.
- `train_cli.py` — Advanced training launcher (Curriculum, WandB).
- `train_ppo.py` — Core PPO training loop and model definitions.
- `watch_agent.py` — Script to load checkpoints and watch agents.
- `run_env.py` — Manual environment verification script.

"""PPO training script for the Wilderness Survival System (WSS) using Gymnasium.

Overview
--------
- Vectorized: runs many environments in parallel via Gymnasium's SyncVectorEnv.
- Provides a clean, well-commented PPO baseline with a simple MLP policy.
- TorchRL-ready wrapper for single env remains available, but this trainer uses
  Gymnasium vector envs for speed and simplicity.
- Reward is intentionally left at 0.0 in the environment. See the commented
  block in :meth:`src.envs.wss_env.WildernessSurvivalEnv.step` to define your
  own reward shaping.
- Includes a simple curriculum: start on ``easy``. If success rate across the
  last ``min_history`` episodes (default 30) reaches ``80%``, increase
  difficulty; if it drops below ``30%``, decrease difficulty.

You can tweak the network, PPO hyperparameters, vectorization degree, and
curriculum thresholds in :func:`train` below.
"""

import sys
import os
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb
except ImportError:
    wandb = None

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.wss_env import WildernessSurvivalEnv
from src.envs.wss_vec_env import WSSNativeVecEnv
from src import Difficulty
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import functools


class WSSObservationWrapper(gym.ObservationWrapper):
    """Flattens and processes the WSS dict observation for MLP policies.
    
    - Terrain (11x11 int): One-hot encoded (6 categories: Fog + 5 types).
    - Items (11x11 int): One-hot encoded (5 categories: None + 4 types).
    - Status (4 int): Normalized by 100.
    - Trader (4x4 int): Scaled by 0.1.
    """
    def __init__(self, env):
        super().__init__(env)
        # Terrain: 11x11=121. 6 categories (0..5). Size 726.
        # Items: 11x11=121. 5 categories (0..4). Size 605.
        # Status: 4.
        # Trader: 16.
        # Prev Action: 13 (One-hot).
        # Total: 1364.
        self.flat_dim = (11 * 11 * 6) + (11 * 11 * 5) + 4 + 16 + 13
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.flat_dim,), dtype=np.float32
        )

    @property
    def difficulty(self):
        return self.env.difficulty

    @difficulty.setter
    def difficulty(self, value):
        self.env.difficulty = value

    def observation(self, obs):
        # 1. Terrain One-Hot
        t = obs["terrain"].flatten()  # (121,)
        t_oh = np.eye(6)[t].flatten() # (726,)

        # 2. Items One-Hot
        i = obs["items"].flatten()    # (121,)
        i_oh = np.eye(5)[i].flatten() # (605,)

        # 3. Status (Normalized)
        s = obs["status"].astype(np.float32) / 100.0

        # 4. Trader (Scaled)
        tr = obs["trader"].flatten().astype(np.float32) * 0.1

        # 5. Prev Action (One-Hot)
        # prev_action is (1,) int
        pa = obs["prev_action"].item()
        pa_oh = np.eye(13)[pa].flatten()

        return np.concatenate([t_oh, i_oh, s, tr, pa_oh])


def create_wrapped_env(width, height, difficulty, render_mode, env_kwargs):
    """Helper to create and wrap a single env instance (picklable)."""
    env = WildernessSurvivalEnv(
        width=width, height=height, difficulty=difficulty, render_mode=render_mode, **(env_kwargs or {})
    )
    return WSSObservationWrapper(env)


def make_vec_env(num_envs=8, width=None, height=None, difficulty="easy", render_mode=None, vector_mode="sync", device=None, **env_kwargs):
    """Create a vectorized environment with ``num_envs`` parallel copies.

    :param int num_envs: Number of parallel environments to run.
    :param int|None width: Map width. If None, determined by difficulty.
    :param int|None height: Map height. If None, determined by difficulty.
    :param str difficulty: Initial difficulty (all envs start on this level).
    :param str|None render_mode: ``None``, ``"human"``, or ``"rgb_array"``.
    :param str vector_mode: ``"sync"`` (default), ``"async"`` (multiprocessing), or ``"native"`` (GPU).
    :param str|torch.device|None device: Device to run on (only used by native vector env).
    :param dict env_kwargs: Extra keyword args forwarded to ``WildernessSurvivalEnv``.
    :return: A VectorEnv running ``num_envs`` WSS envs.
    :rtype: gymnasium.vector.VectorEnv
    """
    if vector_mode == "native":
        return WSSNativeVecEnv(
            num_envs=num_envs,
            width=width,
            height=height,
            difficulty=difficulty,
            render_mode=render_mode,
            device=device,
            **env_kwargs
        )

    # functools.partial is required for multiprocessing pickle compatibility on Windows
    env_fns = [
        functools.partial(create_wrapped_env, width, height, difficulty, render_mode, env_kwargs)
        for _ in range(int(num_envs))
    ]

    if vector_mode == "async":
        return AsyncVectorEnv(env_fns)
    return SyncVectorEnv(env_fns)


class PolicyNet(nn.Module):
    """Hybrid CNN+MLP policy/value network.

    - Visual Input: Terrain (11x11x6) + Items (11x11x5) -> CNN
    - Vector Input: Status (4) + Trader (16) -> MLP
    - Output:
        - logits over discrete actions
        - state-value estimate ``V(s)``

    :param int obs_dim: Observation dimension (flat size, used for validation/slicing).
    :param int action_dim: Number of discrete actions.
    :param tuple[int,int] hidden_sizes: Sizes of hidden layers for the shared trunk.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(512, 256)) -> None:
        super().__init__()
        
        # 1. Visual Encoder (CNN)
        # Input channels: 6 (Terrain) + 5 (Items) = 11
        # Map size: 11x11
        # Enhanced CNN with more depth and channels
        self.cnn = nn.Sequential(
            nn.Conv2d(11, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 11x11 -> 5x5
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size: 128 * 5 * 5 = 3200
        self.cnn_out_dim = 128 * 5 * 5
        
        # 2. Vector Encoder (MLP)
        # Input: 4 (Status) + 16 (Trader) + 13 (PrevAction) = 33
        self.vec_encoder = nn.Sequential(
            nn.Linear(33, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.vec_out_dim = 128
        
        # 3. Fusion & Trunk
        fusion_dim = self.cnn_out_dim + self.vec_out_dim
        
        layers = []
        last = fusion_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.trunk = nn.Sequential(*layers)
        
        # 4. Heads
        self.policy_head = nn.Linear(last, action_dim)
        self.value_head = nn.Linear(last, 1)

    def forward(self, obs: torch.Tensor):
        # obs: (B, 1364)
        
        B = obs.shape[0]
        
        # Hardcoded slicing based on WSSObservationWrapper
        # Terrain: 0..726
        # Items: 726..1331
        # Status: 1331..1335
        # Trader: 1335..1351
        # PrevAction: 1351..1364
        
        terrain_flat = obs[:, 0:726]
        items_flat = obs[:, 726:1331]
        status = obs[:, 1331:1335]
        trader = obs[:, 1335:1351]
        prev_action = obs[:, 1351:1364]
        
        # Reshape Visuals
        # (B, 121, 6) -> (B, 11, 11, 6) -> (B, 6, 11, 11)
        terrain = terrain_flat.view(B, 11, 11, 6).permute(0, 3, 1, 2)
        items = items_flat.view(B, 11, 11, 5).permute(0, 3, 1, 2)
        
        # Concatenate channels: (B, 11, 11, 11)
        visual_in = torch.cat([terrain, items], dim=1)
        
        # CNN
        vis_feat = self.cnn(visual_in)
        
        # Vector
        vec_in = torch.cat([status, trader, prev_action], dim=1)
        vec_feat = self.vec_encoder(vec_in)
        
        # Fusion
        fusion = torch.cat([vis_feat, vec_feat], dim=1)
        
        # Trunk
        feat = self.trunk(fusion)
        
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        
        return logits, value


@torch.no_grad()
def compute_gae_batched(returns_buf, adv_buf, values, rewards, dones, last_values, gamma=0.99, lam=0.95):
    """Compute GAE(lambda) for batched rollouts (T, N).

    :param torch.Tensor returns_buf: Buffer (T, N) to fill with bootstrapped returns.
    :param torch.Tensor adv_buf: Buffer (T, N) to fill with advantages (unnormalized).
    :param torch.Tensor values: Value estimates ``V(s_t)`` of shape (T, N).
    :param torch.Tensor rewards: Rewards ``r_t`` of shape (T, N).
    :param torch.Tensor dones: Terminal flags of shape (T, N) with 1.0 on terminal steps.
    :param torch.Tensor last_values: Bootstrap values ``V(s_T)`` of shape (N,).
    :param float gamma: Discount factor.
    :param float lam: GAE lambda parameter.
    :return: None; buffers are written in-place.
    """
    T, N = rewards.shape
    gae = torch.zeros((N,), dtype=torch.float32, device=rewards.device)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]  # (N,)
        next_value = torch.where(
            torch.ones_like(nonterminal).bool(),  # always true, just to broadcast
            last_values if t == T - 1 else values[t + 1],
            last_values if t == T - 1 else values[t + 1],
        )
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv_buf[t] = gae
        returns_buf[t] = gae + values[t]


class CurriculumManager:
    """Simple difficulty curriculum controller.

    Tracks recent episode outcomes and increases or decreases difficulty when
    thresholds are met. Difficulty order is fixed as
    ``["easy", "medium", "hard", "extreme"]``.

    :param float up_threshold: Success rate required to increase difficulty.
    :param float down_threshold: Failure threshold to decrease difficulty.
    :param int min_history: Minimum number of finished episodes before a change.
    :param str start: Starting difficulty.
    """

    ORDER = ["easy", "medium", "hard", "extreme"]

    def __init__(self, up_threshold=0.8, down_threshold=0.3, min_history=30, start="easy", max_difficulty="extreme"):
        self.up = float(up_threshold)
        self.down = float(down_threshold)
        self.min_history = int(min_history)
        self.max_difficulty = max_difficulty
        # Running history window
        self.history = deque(maxlen=self.min_history)
        self.idx = max(0, min(len(self.ORDER) - 1, self.ORDER.index(start) if start in self.ORDER else 0))

    @property
    def difficulty(self) -> str:
        return self.ORDER[self.idx]

    def add_results(self, successes: list[bool]) -> None:
        """Update running history."""
        self.history.extend(successes)

    def _rate(self) -> float:
        if len(self.history) == 0:
            return 0.0
        return sum(self.history) / len(self.history)

    def maybe_update(self) -> tuple[bool, str, float]:
        """Update difficulty if thresholds trigger.

        :return: (changed, new_difficulty, success_rate)
        """
        if len(self.history) < self.min_history:
            return False, self.difficulty, self._rate()
        rate = self._rate()
        changed = False

        # Check cap
        try:
            max_idx = self.ORDER.index(self.max_difficulty)
        except ValueError:
            max_idx = len(self.ORDER) - 1

        if rate >= self.up and self.idx < len(self.ORDER) - 1 and self.idx < max_idx:
            self.idx += 1
            changed = True
            self.history.clear()
        return changed, self.difficulty, rate


def ppo_update(model, optimizer, obs, actions, old_logp, returns, advantages,
               clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01, epochs=4, batch_size=128):
    """Run multiple epochs of PPO updates.

    :param PolicyNet model: Policy/value network.
    :param torch.optim.Optimizer optimizer: Optimizer instance.
    :param torch.Tensor obs: Observations (N, obs_dim).
    :param torch.Tensor actions: Integer actions (N,).
    :param torch.Tensor old_logp: Log-probs of actions under behavior policy (N,).
    :param torch.Tensor returns: Target returns (N,).
    :param torch.Tensor advantages: Advantage estimates (N,).
    :param float clip_ratio: PPO clip epsilon.
    :param float vf_coef: Value loss coefficient.
    :param float ent_coef: Entropy bonus coefficient.
    :param int epochs: Number of epochs over the dataset per update.
    :param int batch_size: Minibatch size.
    :return: (policy_loss, value_loss, entropy, approx_kl)
    :rtype: tuple[float, float, float, float]
    """
    dataset_size = obs.size(0)
    inds = np.arange(dataset_size)

    # Track average losses
    clip_fracs = []
    approx_kls = []
    pi_losses = []
    v_losses = []
    ents = []

    for _ in range(epochs):
        np.random.shuffle(inds)
        for start in range(0, dataset_size, batch_size):
            idx = inds[start:start + batch_size]
            b_obs = obs[idx]
            b_act = actions[idx]
            b_old_logp = old_logp[idx]
            b_ret = returns[idx]
            b_adv = advantages[idx]

            logits, values = model(b_obs)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(b_act)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - b_old_logp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (b_ret - values).pow(2).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Logging stats
            with torch.no_grad():
                log_ratio = logp - b_old_logp
                approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean() # http://joschu.net/blog/kl-approx.html
                # approx_kl = (b_old_logp - logp).mean()
                approx_kls.append(approx_kl.item())
                pi_losses.append(policy_loss.item())
                v_losses.append(value_loss.item())
                ents.append(entropy.item())

    return np.mean(pi_losses), np.mean(v_losses), np.mean(ents), np.mean(approx_kls)


def train(
    total_steps=200_000,
    rollout_len=256,
    num_envs=8,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    lr=3e-4,
    vf_coef=0.5,
    ent_coef=0.01,
    epochs=4,
    batch_size=4096,
    lr_scheduler="constant",
    target_kl=0.015,
    width=None,
    height=None,
    difficulty="easy",
    render_mode=None,
    vector_mode="sync",
    device=None,
    success_threshold_up=0.8,
    success_threshold_down=0.3,
    min_history=30,
    max_difficulty="extreme",
    checkpoint_freq: int | None = None,
    load_checkpoint: str | None = None,
    env_kwargs: dict | None = None,
    # WandB
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
):
    """Train a vectorized PPO agent on the WSS environment with curriculum.

    :param int total_steps: Total environment interaction steps across all envs.
    :param int rollout_len: Steps per rollout (trajectory segment) before an update.
    :param int num_envs: Number of parallel envs (vectorization degree).
    :param float gamma: Discount factor.
    :param float gae_lambda: GAE lambda parameter.
    :param float clip_ratio: PPO clipping epsilon.
    :param float lr: Adam learning rate.
    :param float vf_coef: Value loss coefficient.
    :param float ent_coef: Entropy bonus coefficient.
    :param int epochs: PPO epochs per update.
    :param int batch_size: PPO minibatch size (over flattened T*N samples).
    :param int|None width: Map width. If None, difficulty-based.
    :param int|None height: Map height. If None, difficulty-based.
    :param str difficulty: Starting difficulty (curriculum begins here).
    :param str|None render_mode: Render mode (``None``/``human``/``rgb_array``).
    :param str vector_mode: Vectorization mode: ``"sync"`` or ``"async"``.
    :param str|torch.device|None device: Device to use for model and native envs.
    :param float success_threshold_up: Success rate to increase difficulty.
    :param float success_threshold_down: Failure threshold to decrease difficulty.
    :param int min_history: Minimum number of finished episodes before changes.
    :return: None. Prints logs to stdout.
    :rtype: None
    """
    # Resolve device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # WandB Init
    if wandb_project and wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            group=wandb_group,
            tags=wandb_tags,
            config={
                "total_steps": total_steps,
                "rollout_len": rollout_len,
                "num_envs": num_envs,
                "vector_mode": vector_mode,
                "gamma": gamma,
                "lr": lr,
                "lr_scheduler": lr_scheduler,
                "target_kl": target_kl,
                "width": width,
                "height": height,
                "difficulty": difficulty,
                "max_difficulty": max_difficulty,
                "device": str(device),
                "success_threshold_up": success_threshold_up,
                "success_threshold_down": success_threshold_down,
                "flat_obs_dim": 1364,
            }
        )

    # Create vector env
    # Inject gamma into env_kwargs to ensure environment reward shaping matches algo gamma
    if env_kwargs is None:
        env_kwargs = {}
    env_kwargs['gamma'] = gamma

    env = make_vec_env(
        num_envs=num_envs,
        width=width,
        height=height,
        difficulty=difficulty,
        render_mode=render_mode,
        vector_mode=vector_mode,
        device=device,
        **env_kwargs,
    )
    obs, infos = env.reset()

    # Resolve observation/action dims (vector env)
    obs_dim = int(np.prod(env.single_observation_space.shape))
    action_dim = int(env.single_action_space.n)

    # Build policy/value network
    model = PolicyNet(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if load_checkpoint:
        print(f"Loading checkpoint from {load_checkpoint}...")
        ckpt = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Checkpoint loaded. Starting training from scratch (resetting steps/optimizer).")

    # Rollout storage (T, N, ...)
    obs_buf = torch.zeros((rollout_len, num_envs, obs_dim), dtype=torch.float32, device=device)
    act_buf = torch.zeros((rollout_len, num_envs), dtype=torch.long, device=device)
    logp_buf = torch.zeros((rollout_len, num_envs), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((rollout_len, num_envs), dtype=torch.float32, device=device)
    done_buf = torch.zeros((rollout_len, num_envs), dtype=torch.float32, device=device)
    val_buf = torch.zeros((rollout_len, num_envs), dtype=torch.float32, device=device)
    ret_buf = torch.zeros((rollout_len, num_envs), dtype=torch.float32, device=device)
    adv_buf = torch.zeros((rollout_len, num_envs), dtype=torch.float32, device=device)

    # Episode stats per env
    ep_returns = np.zeros((num_envs,), dtype=np.float64)
    ep_lengths = np.zeros((num_envs,), dtype=np.int32)

    # Curriculum manager
    # Adjust min_history if it's too small relative to num_envs
    effective_min_history = max(min_history, num_envs * 5)
    if effective_min_history != min_history:
        print(f"Auto-adjusting curriculum min_history from {min_history} to {effective_min_history} (due to num_envs={num_envs} * 5)")

    curriculum = CurriculumManager(
        up_threshold=success_threshold_up,
        down_threshold=success_threshold_down,
        min_history=effective_min_history,
        start=difficulty,
        max_difficulty=max_difficulty,
    )

    ep_ret_queue = deque(maxlen=100)
    ep_len_queue = deque(maxlen=100)
    start_time = time.time()

    step_count = 0
    update_idx = 0
    
    # Checkpointing Init
    next_checkpoint = checkpoint_freq if (checkpoint_freq is not None and checkpoint_freq > 0) else float('inf')
    
    while step_count < total_steps:
        # Track episode stats for this update
        batch_ep_returns = []
        batch_ep_lengths = []
        batch_ep_successes = []

        # Collect a rollout across vectorized envs
        for t in range(rollout_len):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(num_envs, -1)
                logits, values = model(obs_t)  # (N, A), (N,)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()  # (N,)
                logp = dist.log_prob(actions)  # (N,)

            next_obs, rewards, terminateds, truncateds, infos = env.step(actions.cpu().numpy())

            # Handle Tensor/Numpy compatibility
            if isinstance(next_obs, torch.Tensor):
                next_obs = next_obs.cpu().numpy()
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.cpu().numpy()
            if isinstance(terminateds, torch.Tensor):
                terminateds = terminateds.cpu().numpy()
            if isinstance(truncateds, torch.Tensor):
                truncateds = truncateds.cpu().numpy()

            dones = np.logical_or(terminateds, truncateds)

            # Store step data
            obs_buf[t] = obs_t
            act_buf[t] = actions
            logp_buf[t] = logp
            rew_buf[t] = torch.tensor(rewards, dtype=torch.float32, device=device)
            done_buf[t] = torch.tensor(dones.astype(np.float32), dtype=torch.float32, device=device)
            val_buf[t] = values

            # Per-episode accounting
            ep_returns += rewards
            ep_lengths += 1

            # Handle episode ends and gather curriculum results
            finished_idx = np.where(dones)[0]
            finished_successes = []
            if finished_idx.size > 0:
                # Try Gymnasium's final_info channel first
                finfos = infos.get("final_info", None) if isinstance(infos, dict) else None
                for i in finished_idx:
                    success = False
                    if finfos is not None:
                        fi = finfos[i]
                        if isinstance(fi, dict):
                            success = bool(fi.get("reached_goal", False))
                    # Fallback: if no final_info, try current infos as list of dicts
                    if not success and isinstance(infos, (list, tuple)) and i < len(infos) and isinstance(infos[i], dict):
                        success = bool(infos[i].get("reached_goal", False))
                    finished_successes.append(success)
                    # Logging
                    # print(f"Episode[{i}] finished: return={ep_returns[i]:.3f}, length={int(ep_lengths[i])}, success={success}")
                    batch_ep_returns.append(ep_returns[i])
                    batch_ep_lengths.append(ep_lengths[i])
                    batch_ep_successes.append(success)

                    ep_ret_queue.append(ep_returns[i])
                    ep_len_queue.append(ep_lengths[i])
                # Feed curriculum with results
                curriculum.add_results(finished_successes)

                # Maybe change difficulty BEFORE resetting finished envs so the new
                # difficulty applies immediately on reset.
                changed, new_diff, rate = curriculum.maybe_update()
                if changed:
                    print(f"[Curriculum] Changing difficulty to '{new_diff}' (success rate={rate*100:.1f}%).")
                    try:
                        env.set_attr("difficulty", new_diff)
                    except Exception as e:
                        print(f"[WARN] Failed to set difficulty on all envs: {e}")

                # Clear finished episode stats
                ep_returns[finished_idx] = 0.0
                ep_lengths[finished_idx] = 0

            obs = next_obs
            step_count += num_envs

        # Bootstrap value after the last step for each env
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(num_envs, -1)
            _, last_values = model(obs_t)  # (N,)
            compute_gae_batched(
                ret_buf, adv_buf, val_buf, rew_buf, done_buf, last_values,
                gamma=gamma, lam=gae_lambda,
            )
            # Normalize advantages across the whole batch
            adv_mean = adv_buf.mean()
            adv_std = adv_buf.std().clamp_min(1e-8)
            adv_buf = (adv_buf - adv_mean) / adv_std

        # Flatten trajectories (T, N, ...) -> (T*N, ...)
        T, N = rollout_len, num_envs
        flat = lambda x: x.reshape(T * N, *x.shape[2:]) if x.dim() > 2 else x.reshape(T * N)
        obs_flat = flat(obs_buf)
        act_flat = flat(act_buf)
        logp_flat = flat(logp_buf)
        ret_flat = flat(ret_buf)
        adv_flat = flat(adv_buf)

        # PPO update
        pi_loss, v_loss, ent, kl = ppo_update(
            model, optimizer,
            obs=obs_flat, actions=act_flat, old_logp=logp_flat,
            returns=ret_flat, advantages=adv_flat,
            clip_ratio=clip_ratio, vf_coef=vf_coef, ent_coef=ent_coef,
            epochs=epochs, batch_size=batch_size,
        )

        # Learning Rate Scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if lr_scheduler == "linear":
            # Linear decay from initial lr to 0
            frac = 1.0 - (step_count / total_steps)
            new_lr = lr * frac
            new_lr = max(new_lr, 1e-8)
            optimizer.param_groups[0]["lr"] = new_lr
        elif lr_scheduler == "kl_adaptive":
            # Adaptive based on Target KL
            if kl > target_kl * 2.0:
                optimizer.param_groups[0]["lr"] = max(current_lr / 1.5, 1e-6)
            elif kl < target_kl / 2.0:
                optimizer.param_groups[0]["lr"] = min(current_lr * 1.5, 1e-2)

        update_idx += 1

        # --- Console Logging ---
        if len(batch_ep_returns) > 0:
            mean_ret = np.mean(batch_ep_returns)
            mean_len = np.mean(batch_ep_lengths)
            success_rate = np.mean(batch_ep_successes) * 100.0
            print(f"Update {update_idx} (Step {step_count}): "
                  f"Episodes={len(batch_ep_returns)}, "
                  f"Mean Return={mean_ret:.3f}, "
                  f"Mean Length={mean_len:.1f}, "
                  f"Success={success_rate:.1f}%")
        else:
            print(f"Update {update_idx} (Step {step_count}): No episodes finished this batch.")
        
        print(f"    Losses: pi={pi_loss:.3f}, v={v_loss:.3f}, ent={ent:.3f}, kl={kl:.4f}")
        
        # Checkpointing
        if step_count >= next_checkpoint:
            run_id = wandb_name if wandb_name else "manual_run"
            ckpt_dir = os.path.join("checkpoints", run_id)
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{step_count}.pt")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step_count,
                'difficulty': curriculum.difficulty,
                'curriculum_state': {
                    'idx': curriculum.idx,
                    'history': list(curriculum.history),
                },
                'config': env_kwargs
            }, ckpt_path)
            print(f"[Checkpoint] Saved model to {ckpt_path}")
            next_checkpoint += checkpoint_freq

        # -----------------------

        # Log to WandB
        if wandb_project and wandb:
            avg_ret = np.mean(ep_ret_queue) if ep_ret_queue else 0.0
            avg_len = np.mean(ep_len_queue) if ep_len_queue else 0.0
            sps = int(step_count / (time.time() - start_time))
            hardness = Difficulty.get_hardness(curriculum.difficulty) if Difficulty else 0.0

            wandb.log({
                "losses/policy_loss": pi_loss,
                "losses/value_loss": v_loss,
                "losses/entropy": ent,
                "losses/approx_kl": kl,
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "curriculum/difficulty": curriculum.difficulty,
                "curriculum/success_rate": curriculum._rate(),
                "curriculum/hardness": hardness,
                "charts/episodic_return": avg_ret,
                "charts/episodic_length": avg_len,
                "charts/SPS": sps,
                "global_step": step_count,
            }, step=step_count)

        if update_idx % 10 == 0:
            print(f"Update {update_idx}: difficulty='{curriculum.difficulty}', steps so far={step_count}")

    if wandb_project and wandb:
        wandb.finish()
    print("Training finished.")


if __name__ == "__main__":
    # Quick defaults; edit as needed.
    train(
        total_steps=200_000,
        rollout_len=256,
        num_envs=8,
        width=None,
        height=None,
        difficulty="easy",  # curriculum starts here
        render_mode=None,  # set to "human" to watch; training is faster with None
    )

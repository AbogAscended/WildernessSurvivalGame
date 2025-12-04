"""Command-line launcher for vectorized PPO training on WSS with curriculum.

This script wraps :func:`train` from ``train_ppo.py`` and exposes most
environment and PPO hyperparameters via argparse. It also forwards extra
environment options (e.g., renderer toggles) to the env constructor.
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_ppo import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on WSS (vectorized, with curriculum).")
    # PPO / rollout
    p.add_argument('--total-steps', type=int, default=900_000_000)
    p.add_argument('--rollout-len', type=int, default=256)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip-ratio', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=0.00002601229487374892)
    p.add_argument('--vf-coef', type=float, default=0.5)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--lr-scheduler', type=str, default='kl_adaptive', choices=['constant', 'linear', 'kl_adaptive'], help='Learning rate scheduler.')
    p.add_argument('--target-kl', type=float, default=0.03, help='Target KL divergence for adaptive scheduler.')

    # Env
    p.add_argument('--difficulty', type=str, default='medium', choices=['easy', 'medium', 'hard', 'extreme', 'normal'])
    p.add_argument('--width', type=int, default=None, help='Map width (optional, overrides difficulty default).')
    p.add_argument('--height', type=int, default=None, help='Map height (optional, overrides difficulty default).')
    p.add_argument('--render-mode', type=str, default='human', choices=[None, 'human', 'rgb_array', 'pygame'], help='Rendering mode for envs.')
    p.add_argument('--vector-mode', type=str, default='native', choices=['sync', 'native'], help='Vectorization mode (sync=serial, native=gpu).')
    p.add_argument('--device', type=str, default="cuda", help='Device to run on (e.g. cpu, cuda). If None, auto-detects.')
    p.add_argument('--window-radius', type=int, default=2)
    p.add_argument('--max-steps', type=int, default=None)

    # Curriculum
    p.add_argument('--success-thresh-up', type=float, default=0.8)
    p.add_argument('--success-thresh-down', type=float, default=0.3)
    p.add_argument('--min-history', type=int, default=30)
    p.add_argument('--max-difficulty', type=str, default='medium', choices=['easy', 'medium', 'hard', 'extreme'], help='Maximum difficulty level for curriculum.')

    # ASCII renderer toggles
    p.add_argument('--legend', dest='legend', action='store_true', default=True, help='Show legend (default on).')
    p.add_argument('--no-legend', dest='legend', action='store_false', help='Hide legend.')
    p.add_argument('--ascii-colors', action='store_true', help='Enable ANSI colors in ASCII (human mode).')
    p.add_argument('--clear-screen', action='store_true', help='Clear screen each frame in ASCII (human mode).')

    # Checkpointing
    p.add_argument('--checkpoint-freq', type=int, default=1_000_000, help='Save model checkpoint every N steps.')
    p.add_argument('--load-checkpoint', type=str, default=None, help='Path to checkpoint to load (train from scratch with these weights).')

    # WandB
    p.add_argument('--wandb-project', type=str, default='WildernessSurvival', help='WandB project name')
    p.add_argument('--wandb-entity', type=str, default=None, help='WandB entity (team/user)')
    p.add_argument('--wandb-name', type=str, default="WSSRLRun", help='WandB run name')
    p.add_argument('--wandb-group', type=str, default=None, help='WandB run group')
    p.add_argument('--wandb-tags', type=str, nargs='*', default=None, help='WandB tags')

    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Normalize difficulty alias
    diff = 'medium' if args.difficulty == 'normal' else args.difficulty

    env_kwargs = {
        'window_radius': args.window_radius,
        'max_steps': args.max_steps,
        'show_legend': args.legend,
        'ascii_colors': args.ascii_colors,
        'clear_screen': args.clear_screen,
    }

    print(
        f"Starting PPO training: total_steps={args.total_steps}, rollout_len={args.rollout_len}, "
        f"num_envs={args.num_envs}, difficulty={diff}, render_mode={args.render_mode}"
    )

    train(
        total_steps=args.total_steps,
        rollout_len=args.rollout_len,
        num_envs=args.num_envs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        lr=args.lr,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_scheduler=args.lr_scheduler,
        target_kl=args.target_kl,
        width=args.width,
        height=args.height,
        difficulty=diff,
        render_mode=args.render_mode,
        vector_mode=args.vector_mode,
        device=args.device,
        success_threshold_up=args.success_thresh_up,
        success_threshold_down=args.success_thresh_down,
        min_history=args.min_history,
        max_difficulty=args.max_difficulty,
        checkpoint_freq=args.checkpoint_freq,
        load_checkpoint=args.load_checkpoint,
        env_kwargs=env_kwargs,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_group=args.wandb_group,
        wandb_tags=args.wandb_tags,
    )


if __name__ == '__main__':
    main()

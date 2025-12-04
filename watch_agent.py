import sys
import time
import argparse
from pathlib import Path
import torch
import numpy as np

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_ppo import make_vec_env, PolicyNet

def parse_args():
    p = argparse.ArgumentParser(description="Watch a trained agent play WSS.")
    p.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint.')
    p.add_argument('--delay', type=float, default=0.5, help='Delay between steps in seconds.')
    p.add_argument('--width', type=int, default=None)
    p.add_argument('--height', type=int, default=None)
    p.add_argument('--difficulty', type=str, default='medium')
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--ascii-colors', action='store_true')
    p.add_argument('--clear-screen', action='store_true')
    p.add_argument('--render-mode', type=str, default='human', choices=['human', 'pygame', 'rgb_array'])
    p.add_argument('--episodes', type=int, default=1, help='Number of episodes to watch.')
    return p.parse_args()

def main():
    args = parse_args()
    
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    env_kwargs = {
        'max_steps': args.max_steps,
        'ascii_colors': args.ascii_colors,
        'clear_screen': args.clear_screen,
        'show_legend': True,
    }
    
    # Create single vectorized env to match training dimensions
    env = make_vec_env(
        num_envs=1,
        width=args.width,
        height=args.height,
        difficulty=args.difficulty,
        render_mode=args.render_mode,
        vector_mode="sync", 
        device=device,
        **env_kwargs
    )

    obs_dim = int(np.prod(env.single_observation_space.shape))
    action_dim = int(env.single_action_space.n)

    model = PolicyNet(obs_dim, action_dim).to(device)
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Starting watch for {args.episodes} episodes...")
    
    for ep in range(args.episodes):
        obs, info = env.reset()
        
        done = False
        total_reward = 0
        steps = 0
        
        print(f"Episode {ep+1}/{args.episodes} started.")
        
        try:
            while not done:
                # Convert obs to tensor
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    logits, _ = model(obs_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                
                # Action to numpy for env step (if needed, but vec env usually accepts numpy or int)
                action_np = action.cpu().numpy()
                
                obs, reward, terminated, truncated, info = env.step(action_np)
                
                total_reward += reward.item()
                steps += 1
                
                time.sleep(args.delay)
                
                if terminated[0] or truncated[0]:
                    done = True
                    # Access final info if available for accurate reward/steps logging if needed
                    # but total_reward accumulation above is sufficient for display
                    print(f"Episode {ep+1} finished. Steps: {steps}, Total Reward: {total_reward:.2f}")
                    
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

    env.close()

if __name__ == '__main__':
    main()

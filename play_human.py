"""Interactive human player for the Wilderness Survival System (WSS).

Controls (vi, WASD, diagonals):
- Movement:
  - h/j/k/l = Left/Down/Up/Right
  - w/a/s/d = Up/Left/Down/Right
  - y/u/b/n = Up-Left/Up-Right/Down-Left/Down-Right
  - q/e/z/c = Up-Left/Up-Right/Down-Left/Down-Right
- Wait/Rest: '.' (dot) or Space or 'x'
- Quit: 'Q' (uppercase) or Ctrl-C

Render:
- Uses NetHack-style ASCII with a legend on the top-right by default.
  You can toggle colors and screen clearing via command line flags.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from src.envs.wss_env import WildernessSurvivalEnv
from src import Trader


# Map single-key inputs to env action IDs
KEY_TO_ACTION = {
    # Wait
    '.': 0, ' ': 0, 'x': 0,
    # Trade options
    '1': 91, '2': 92, '3': 93, '4': 94,
    # Cardinal (vi)
    'k': 1,  # North (up)
    'j': 2,  # South (down)
    'l': 3,  # East (right)
    'h': 4,  # West (left)
    # Cardinal (WASD)
    'w': 1,
    's': 2,
    'd': 3,
    'a': 4,
    # Diagonals (vi-like)
    'u': 5,  # NE
    'y': 6,  # NW
    'n': 7,  # SE
    'b': 8,  # SW
    # Diagonals (QEZC)
    'e': 5,  # NE
    'q': 6,  # NW
    'c': 7,  # SE
    'z': 8,  # SW
}


def get_key() -> str:
    """Read a single keypress as a string (case-preserving).

    Uses ``msvcrt.getch`` on Windows for non-blocking single-character input,
    with a portable fallback to ``input()`` on other platforms.
    """
    try:
        import msvcrt  # type: ignore
        ch = msvcrt.getch()
        # Handle arrow keys (two-byte sequences) by ignoring the prefix
        if ch in (b'\x00', b'\xe0'):
            ch = msvcrt.getch()
        try:
            return ch.decode('utf-8', errors='ignore')
        except Exception:
            return ''
    except Exception:
        # Fallback: prompt the user
        s = input("Move [h/j/k/l, y/u/b/n, wasd, q/e/z/c, '.' wait, Q quit]: ").strip()
        return s[:1] if s else ''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play WSS as a human (ASCII or Pygame).")
    p.add_argument('--width', type=int, default=None, help='Map width (columns). Default: random based on difficulty.')
    p.add_argument('--height', type=int, default=None, help='Map height (rows). Default: random based on difficulty.')
    p.add_argument('--difficulty', type=str, default='medium', choices=['easy', 'medium', 'hard', 'extreme', 'normal'], help='Difficulty level (normal aliases to medium).')
    p.add_argument('--max-steps', type=int, default=1000, help='Optional step cap before truncation.')
    # Backend selection
    p.add_argument('--backend', type=str, default='ascii', choices=['ascii', 'pygame'], help='Renderer backend.')
    # ASCII toggles
    p.add_argument('--no-legend', action='store_true', help='Hide legend on the right (ASCII & Pygame).')
    p.add_argument('--colors', action='store_true', help='Enable ANSI colors (ASCII only).')
    p.add_argument('--clear-screen', action='store_true', help='Clear screen each frame (ASCII only).')
    # Pygame options
    p.add_argument('--cell', type=int, default=28, help='Pygame cell size (pixels per tile).')
    p.add_argument('--no-grid', action='store_true', help='Disable Pygame grid overlay.')
    p.add_argument('--legend-width', type=int, default=240, help='Pygame legend panel width (pixels).')
    p.add_argument('--fps', type=int, default=30, help='Pygame target FPS.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    diff = 'medium' if args.difficulty == 'normal' else args.difficulty
    # Choose render backend based on --backend (avoid creating a Pygame window when using ASCII)
    render_mode = 'pygame' if args.backend == 'pygame' else 'human'
    env_kwargs = dict(
        width=args.width,
        height=args.height,
        difficulty=diff,
        render_mode=render_mode,
        max_steps=args.max_steps,
        show_legend=not args.no_legend,
    )
    if args.backend == 'ascii':
        env_kwargs.update(dict(ascii_colors=args.colors, clear_screen=args.clear_screen))
    else:
        env_kwargs.update(dict(
            pygame_cell_size=args.cell,
            pygame_show_grid=(not args.no_grid),
            pygame_legend_width=args.legend_width,
            pygame_fps=args.fps,
        ))

    env = WildernessSurvivalEnv(**env_kwargs)

    obs, info = env.reset()

    done = False
    truncated = False
    total_reward = 0.0
    steps = 0

    try:
        if args.backend == 'pygame':
            # Pygame interactive loop: continuously render and poll keyboard state
            # so the window stays responsive even when the user is thinking.
            import pygame

            def pick_action_from_keys() -> int | None:
                # Ensure the event queue is pumped before sampling keys to keep the window responsive
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                mods = pygame.key.get_mods()
                # Quit conditions: ESC or Shift+Q (uppercase Q). Plain 'q' remains a move (NW).
                if keys[pygame.K_ESCAPE] or (keys[pygame.K_q] and (mods & pygame.KMOD_SHIFT)):
                    return -1  # sentinel for quit
                # Diagonals first (single key variants)
                if keys[pygame.K_u] or keys[pygame.K_e]:
                    return 5  # NE
                if keys[pygame.K_y] or (keys[pygame.K_q] and not (mods & pygame.KMOD_SHIFT)):
                    return 6  # NW
                if keys[pygame.K_n] or keys[pygame.K_c]:
                    return 7  # SE
                if keys[pygame.K_b] or keys[pygame.K_z]:
                    return 8  # SW
                # Cardinals (arrows, WASD, vi)
                if keys[pygame.K_UP] or keys[pygame.K_k] or keys[pygame.K_w]:
                    return 1  # North
                if keys[pygame.K_DOWN] or keys[pygame.K_j] or keys[pygame.K_s]:
                    return 2  # South
                if keys[pygame.K_RIGHT] or keys[pygame.K_l] or keys[pygame.K_d]:
                    return 3  # East
                if keys[pygame.K_LEFT] or keys[pygame.K_h] or keys[pygame.K_a]:
                    return 4  # West
                # Wait / rest
                if keys[pygame.K_PERIOD] or keys[pygame.K_SPACE] or keys[pygame.K_x]:
                    return 0
                # Trade options
                if keys[pygame.K_1]: return 91
                if keys[pygame.K_2]: return 92
                if keys[pygame.K_3]: return 93
                if keys[pygame.K_4]: return 94
                return None

            while not (done or truncated):
                # Render a frame (also pumps the pygame event queue inside renderer)
                env.render()

                # Check for trader and show overlay
                if env.player and env.map:
                    px, py = env.player.position
                    if 0 <= px < env.map.width and 0 <= py < env.map.height:
                        tile = env.map.getTile(px, py)
                        if tile.has_trader():
                            trader = next((i for i in tile.items if isinstance(i, Trader.Trader)), None)
                            if trader:
                                if hasattr(env, "renderer") and hasattr(env.renderer, "draw_trading_menu"):
                                    env.renderer.draw_trading_menu(trader)
                                elif hasattr(env, "renderer") and hasattr(env.renderer, "draw_overlay_message"):
                                    env.renderer.draw_overlay_message("Trader encountered! Options 1-4")

                # Additionally handle a bare QUIT event here so window closes immediately
                quit_now = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_now = True
                if quit_now:
                    print("Quitting.")
                    break

                act = pick_action_from_keys()
                if act is None:
                    continue  # no input this frame; keep rendering
                if act == -1:
                    print("Quitting.")
                    break
                
                # Handle trade actions
                if act >= 90:
                    if env.player and env.map:
                        px, py = env.player.position
                        tile = env.map.getTile(px, py)
                        if tile.has_trader():
                            trader = next((i for i in tile.items if isinstance(i, Trader.Trader)), None)
                            if trader:
                                idx = act - 91
                                success = env.player.execute_trade(trader, idx)
                                if success:
                                    print(f"Trade option {idx+1} successful!")
                                else:
                                    print(f"Trade option {idx+1} failed (insufficient funds or invalid option).")
                    # After attempting trade, we treat it as a wait step
                    act = 0

                obs, reward, done, truncated, info = env.step(int(act))
                total_reward += float(reward)
                steps += 1
            print(f"Episode finished in {steps} steps. Total reward: {total_reward:.3f}. Info: {info}")
        else:
            # ASCII / terminal loop uses single-key input helper
            while not (done or truncated):
                # Check for trader
                if env.player and env.map:
                    px, py = env.player.position
                    tile = env.map.getTile(px, py)
                    if tile.has_trader():
                        print("Trader encountered! Press 1-4 to trade.")

                key = get_key()
                if not key:
                    continue
                if key in ('Q', '\x1b'):
                    print("Quitting.")
                    break
                action = KEY_TO_ACTION.get(key.lower(), None)
                if action is None:
                    continue
                
                act = int(action)
                # Handle trade actions
                if act >= 90:
                    if env.player and env.map:
                        px, py = env.player.position
                        tile = env.map.getTile(px, py)
                        if tile.has_trader():
                            trader = next((i for i in tile.items if isinstance(i, Trader.Trader)), None)
                            if trader:
                                idx = act - 91
                                success = env.player.execute_trade(trader, idx)
                                if success:
                                    print(f"Trade option {idx+1} successful!")
                                else:
                                    print(f"Trade option {idx+1} failed.")
                    act = 0

                obs, reward, done, truncated, info = env.step(act)
                total_reward += float(reward)
                steps += 1
            print(f"Episode finished in {steps} steps. Total reward: {total_reward:.3f}. Info: {info}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()

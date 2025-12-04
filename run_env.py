"""Manual runner for the Wilderness Survival System (WSS) environment.

Prompts the user for map size, difficulty, and render mode, then runs a single
episode using a random policy. Useful for quick sanity checks and visualizing
the ASCII/RGB rendering.
"""

import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.wss_env import WildernessSurvivalEnv


def prompt_int(prompt: str, default: int) -> int:
    """Prompt the user for an integer with a default fallback.

    :param str prompt: Message displayed to the user.
    :param int default: Value used when input is empty or invalid.
    :return: Parsed integer or the default value.
    :rtype: int
    """
    try:
        s = input(f"{prompt} [{default}]: ").strip()
        return int(s) if s else default
    except Exception:
        return default


def prompt_choice(prompt: str, choices: list, default: str) -> str:
    """Prompt the user to choose from a list of options.

    :param str prompt: Message displayed to the user.
    :param list choices: Permitted string choices.
    :param str default: Default value returned on empty/invalid input.
    :return: The chosen string from ``choices`` or the default.
    :rtype: str
    """
    s = input(f"{prompt} {choices} [{default}]: ").strip()
    return s if s in choices else default


def main():
    """Run one random-policy episode with interactive configuration.

    Prompts for map size, difficulty (accepting ``normal`` as alias for
    ``medium``), and render mode, then executes the episode while accumulating
    total reward. Prints a summary upon completion.
    """
    print("Wilderness Survival System (WSS) — quick run")
    width = prompt_int("Map width", 20)
    height = prompt_int("Map height", 10)
    difficulty = prompt_choice("Difficulty", ["easy", "medium", "hard", "extreme", "normal"], "medium")
    # Backward-compat alias
    if difficulty == "normal":
        difficulty = "medium"
    render = prompt_choice("Render mode", ["none", "human", "rgb_array", "pygame"], "human")
    render_mode = None if render == "none" else render

    env = WildernessSurvivalEnv(width=width, height=height, difficulty=difficulty, render_mode=render_mode, max_steps=1000)
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0
    # simple random policy for demonstration
    import numpy as np
    while not (done or truncated):
        action = int(np.random.randint(env.action_space.n))
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
    print(f"Episode finished in {steps} steps. Total reward: {total_reward:.3f}. Info: {info}")


if __name__ == "__main__":
    main()

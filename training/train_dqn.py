"""
DQN Training Pipeline for Smart Energy Grid Optimization
=========================================================
Training script for the Deep Q-Network agent on the SmartGridEnv.
Includes logging, checkpointing, and evaluation.

Usage:
    python training/train_dqn.py --config configs/dqn_config.yaml --episodes 1000

Author: Pranay M Mahendrakar
"""

import os
import sys
import argparse
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.grid_env import SmartGridEnv
from agents.dqn_agent import DQNAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN agent on Smart Grid environment")
    parser.add_argument("--config", type=str, default="configs/dqn_config.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes (overrides config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/dqn",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Interval for logging training metrics")
    parser.add_argument("--eval_interval", type=int, default=200,
                        help="Interval for running evaluation episodes")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate_agent(agent: DQNAgent, env: SmartGridEnv, n_episodes: int = 10) -> dict:
    """Run evaluation episodes and return metrics."""
    episode_rewards = []
    power_losses = []
    renewable_utils = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        ep_loss = []
        ep_renew = []
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(obs, training=False)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            ep_loss.append(info.get("power_loss", 0))
            ep_renew.append(info.get("renewable_utilization", 0))

        episode_rewards.append(total_reward)
        power_losses.append(np.mean(ep_loss))
        renewable_utils.append(np.mean(ep_renew))

    return {
        "eval_mean_reward": float(np.mean(episode_rewards)),
        "eval_std_reward": float(np.std(episode_rewards)),
        "eval_mean_power_loss": float(np.mean(power_losses)),
        "eval_mean_renewable_util": float(np.mean(renewable_utils)),
    }


def train(config: dict, args: argparse.Namespace):
    """Main training loop."""

    # Setup
    np.random.seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DQN Training — Smart Energy Grid Optimization")
    print(f"  Timestamp: {timestamp}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"{'='*60}\n")

    # Environment setup
    env_cfg = config.get("environment", {})
    env = SmartGridEnv(
        n_nodes=env_cfg.get("n_nodes", 14),
        n_lines=env_cfg.get("n_lines", 20),
        n_storage=env_cfg.get("n_storage", 3),
        max_steps=env_cfg.get("max_steps", 168),
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = config.get("agent", {}).get("action_bins", 10)  # Discretized actions

    print(f"Environment: {obs_dim}D observation, {act_dim} discrete actions")

    # Agent setup
    agent_cfg = config.get("agent", {})
    agent = DQNAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        learning_rate=agent_cfg.get("learning_rate", 1e-4),
        gamma=agent_cfg.get("gamma", 0.99),
        epsilon_start=agent_cfg.get("epsilon_start", 1.0),
        epsilon_end=agent_cfg.get("epsilon_end", 0.05),
        epsilon_decay=agent_cfg.get("epsilon_decay", 50_000),
        buffer_capacity=agent_cfg.get("buffer_capacity", 100_000),
        batch_size=agent_cfg.get("batch_size", 256),
        target_update_freq=agent_cfg.get("target_update_freq", 1000),
        hidden_dims=agent_cfg.get("hidden_dims", [256, 256, 128]),
    )

    # Training config
    train_cfg = config.get("training", {})
    n_episodes = args.episodes or train_cfg.get("n_episodes", 1000)
    warmup_steps = train_cfg.get("warmup_steps", 5000)

    print(f"Training for {n_episodes} episodes")
    print(f"Warmup steps: {warmup_steps}\n")

    # Tracking
    best_eval_reward = float("-inf")
    episode_rewards = []
    recent_rewards = []

    # --- Warmup: fill replay buffer with random experience ---
    print("Filling replay buffer with random experience...")
    obs, _ = env.reset(seed=args.seed)
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        # Map continuous action to discrete index for DQN
        discrete_action = np.random.randint(act_dim)
        next_obs, reward, done, truncated, info = env.step(
            _continuous_from_discrete(discrete_action, act_dim, env.action_space)
        )
        agent.push_experience(obs, discrete_action, reward, next_obs, float(done or truncated))
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()
    print(f"Replay buffer size: {len(agent.replay_buffer)}\n")

    # --- Main training loop ---
    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        ep_losses = []
        done = False
        truncated = False

        while not (done or truncated):
            discrete_action = agent.select_action(obs, training=True)
            continuous_action = _continuous_from_discrete(discrete_action, act_dim, env.action_space)

            next_obs, reward, done, truncated, info = env.step(continuous_action)
            agent.push_experience(obs, discrete_action, reward, next_obs, float(done or truncated))

            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)

            total_reward += reward
            obs = next_obs

        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        # Logging
        if episode % args.log_interval == 0:
            stats = agent.get_stats()
            avg_loss = np.mean(ep_losses) if ep_losses else 0
            print(
                f"Episode {episode:5d}/{n_episodes} | "
                f"Reward: {total_reward:8.2f} | "
                f"Avg(100): {np.mean(recent_rewards):8.2f} | "
                f"Epsilon: {stats['epsilon']:.4f} | "
                f"Loss: {avg_loss:.5f}"
            )

        # Evaluation
        if episode % args.eval_interval == 0:
            eval_metrics = evaluate_agent(agent, env, n_episodes=10)
            print(f"\n  [EVAL] Episode {episode} | "
                  f"Mean Reward: {eval_metrics['eval_mean_reward']:.2f} ± "
                  f"{eval_metrics['eval_std_reward']:.2f} | "
                  f"Power Loss: {eval_metrics['eval_mean_power_loss']:.4f} | "
                  f"Renewable Util: {eval_metrics['eval_mean_renewable_util']:.3f}\n")

            # Save best model
            if eval_metrics["eval_mean_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["eval_mean_reward"]
                agent.save(str(checkpoint_dir / "dqn_best.pth"))
                print(f"  New best model saved! Reward: {best_eval_reward:.2f}\n")

        # Periodic checkpoint
        if episode % 500 == 0:
            agent.save(str(checkpoint_dir / f"dqn_episode_{episode}.pth"))

    # Final save
    agent.save(str(checkpoint_dir / "dqn_final.pth"))
    print(f"\nTraining complete! Best eval reward: {best_eval_reward:.2f}")
    print(f"Models saved to: {checkpoint_dir}")
    env.close()


def _continuous_from_discrete(action_idx: int, n_actions: int, action_space) -> np.ndarray:
    """Map discrete action index to continuous action vector (uniform grid)."""
    # Simple mapping: evenly divide action space
    t = action_idx / max(n_actions - 1, 1)
    low = action_space.low
    high = action_space.high
    return (low + t * (high - low)).astype(np.float32)


if __name__ == "__main__":
    args = parse_args()

    # Load or use default config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file not found: {args.config}. Using defaults.")
        config = {
            "environment": {"n_nodes": 14, "n_lines": 20, "n_storage": 3, "max_steps": 168},
            "agent": {
                "learning_rate": 1e-4,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
                "epsilon_decay": 50000,
                "buffer_capacity": 100000,
                "batch_size": 256,
                "target_update_freq": 1000,
                "hidden_dims": [256, 256, 128],
                "action_bins": 10,
            },
            "training": {"n_episodes": 1000, "warmup_steps": 5000},
        }

    train(config, args)

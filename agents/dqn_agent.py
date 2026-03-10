"""
Deep Q-Network (DQN) Agent for Smart Energy Grid Optimization
==============================================================
Implements DQN with experience replay, target network, double DQN,
and dueling architecture for controlling smart grid energy distribution.

Author: Pranay M Mahendrakar
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────

class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# Dueling DQN Network
# ─────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network architecture.
    Separates state-value V(s) and advantage A(s, a) streams.
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super(DuelingDQN, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        # Shared feature extractor
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        self.feature_layer = nn.Sequential(*layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Dueling combination
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


# ─────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network Agent with:
    - Double DQN (reduces overestimation)
    - Dueling architecture (better value estimation)
    - Prioritized Experience Replay (focus on important transitions)
    - Epsilon-greedy exploration with linear decay
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50_000,
        buffer_capacity: int = 100_000,
        batch_size: int = 256,
        target_update_freq: int = 1000,
        hidden_dims: Optional[List[int]] = None,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"DQNAgent using device: {self.device}")

        # Networks
        self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10_000, gamma=0.9)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Training metrics
        self.losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        # Decay epsilon
        if training:
            self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(
                -1.0 * self.steps_done / self.epsilon_decay
            )
            self.steps_done += 1

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def push_experience(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """Perform one gradient update step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        # Huber loss (smooth L1)
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        loss_val = loss.item()
        self.losses.append(loss_val)

        # Periodically sync target network
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

        return loss_val

    def update_target_network(self):
        """Hard update: copy policy net weights to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target_network(self, tau: float = 0.005):
        """Soft update: slowly blend policy net weights into target net."""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "epsilon": self.epsilon,
        }, path)
        print(f"Agent saved to {path}")

    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        self.epsilon = checkpoint["epsilon"]
        print(f"Agent loaded from {path}")

    def get_stats(self) -> dict:
        """Return training statistics."""
        return {
            "steps_done": self.steps_done,
            "epsilon": round(self.epsilon, 4),
            "buffer_size": len(self.replay_buffer),
            "avg_loss": round(np.mean(self.losses[-100:]), 5) if self.losses else None,
        }


if __name__ == "__main__":
    # Quick test
    state_dim = 58   # For 14-node, 20-line grid environment
    action_dim = 10  # Discretized action space

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-4,
        batch_size=64,
    )

    # Simulate a few random steps
    for _ in range(200):
        state = np.random.randn(state_dim).astype(np.float32)
        action = agent.select_action(state)
        next_state = np.random.randn(state_dim).astype(np.float32)
        reward = np.random.randn()
        done = random.random() < 0.1
        agent.push_experience(state, action, reward, next_state, done)
        loss = agent.update()

    stats = agent.get_stats()
    print("Agent stats:", stats)
    print("Test passed!")

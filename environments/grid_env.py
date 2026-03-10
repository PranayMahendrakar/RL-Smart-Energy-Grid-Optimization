"""
Smart Energy Grid Environment for Reinforcement Learning
=========================================================
A custom OpenAI Gymnasium environment simulating a smart energy grid.
The RL agent controls energy routing, storage dispatch, and demand response
to minimize power losses and maximize grid stability.

Author: Pranay M Mahendrakar
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class SmartGridEnv(gym.Env):
    """
    Smart Energy Grid Environment.

    Observation Space:
        - Load demand at each node (N values, normalized 0-1)
        - Generation at each node (N values, normalized 0-1)
        - Storage state of charge at each node (N values, 0-1)
        - Grid line loading (L values, 0-1)
        - Time of day (1 value, 0-1 normalized)

    Action Space:
        - Continuous: Power flow setpoints for each controllable line
        - Storage charge/discharge commands for each storage unit

    Reward:
        r = -alpha * power_loss - beta * instability_penalty
            + gamma * renewable_utilization + delta * demand_satisfaction
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_nodes: int = 14,
        n_lines: int = 20,
        n_storage: int = 3,
        max_steps: int = 24 * 7,  # One week at hourly resolution
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.n_nodes = n_nodes
        self.n_lines = n_lines
        self.n_storage = n_storage
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Reward shaping weights
        self.alpha = 1.0    # Power loss penalty weight
        self.beta = 0.5     # Instability penalty weight
        self.gamma = 0.3    # Renewable utilization reward weight
        self.delta = 0.8    # Demand satisfaction reward weight

        # === Observation Space ===
        # [load (N), generation (N), storage_soc (N_storage), line_loading (L), time (1)]
        obs_size = n_nodes + n_nodes + n_storage + n_lines + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        # === Action Space ===
        # Continuous actions: [line_setpoints (L), storage_commands (N_storage)]
        # Line setpoints: -1 (max reverse flow) to +1 (max forward flow)
        # Storage: -1 (max discharge) to +1 (max charge)
        action_size = n_lines + n_storage
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(action_size,),
            dtype=np.float32
        )

        # Grid topology (adjacency matrix)
        self.topology = self._create_grid_topology()

        # State variables
        self.current_step = 0
        self.load_profiles = None
        self.renewable_profiles = None
        self.storage_soc = None

        # Grid parameters
        self.line_capacity = np.ones(n_lines) * 100.0   # MVA
        self.storage_capacity = np.ones(n_storage) * 50.0  # MWh
        self.storage_max_power = np.ones(n_storage) * 25.0  # MW

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step = 0

        # Generate synthetic load and renewable profiles
        self.load_profiles = self._generate_load_profiles()
        self.renewable_profiles = self._generate_renewable_profiles()

        # Initialize storage at 50% state of charge
        self.storage_soc = np.full(self.n_storage, 0.5)

        observation = self._get_observation()
        info = {"step": 0, "total_loss": 0.0}

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment."""

        # Parse actions
        line_setpoints = action[:self.n_lines]
        storage_commands = action[self.n_lines:]

        # Update storage state of charge
        self._update_storage(storage_commands)

        # Simulate power flow with given setpoints
        power_flow_result = self._simulate_power_flow(line_setpoints)

        # Calculate reward components
        power_loss = power_flow_result["total_loss"]
        instability = power_flow_result["voltage_deviation"]
        renewable_utilization = power_flow_result["renewable_utilization"]
        demand_satisfaction = power_flow_result["demand_satisfaction"]

        reward = (
            -self.alpha * power_loss
            - self.beta * instability
            + self.gamma * renewable_utilization
            + self.delta * demand_satisfaction
        )

        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.max_steps

        # Check for grid collapse (instability > threshold)
        if instability > 0.3:
            terminated = True
            reward -= 10.0  # Large penalty for grid collapse

        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "power_loss": float(power_loss),
            "instability": float(instability),
            "renewable_utilization": float(renewable_utilization),
            "demand_satisfaction": float(demand_satisfaction),
        }

        return observation, float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector from current grid state."""
        t = self.current_step % len(self.load_profiles[0])

        load = self.load_profiles[:, t] / self.load_profiles.max()
        generation = self.renewable_profiles[:, t] / (self.renewable_profiles.max() + 1e-8)
        soc = self.storage_soc.copy()
        line_loading = np.random.uniform(0.2, 0.8, self.n_lines)  # Placeholder
        time_feature = np.array([(t % 24) / 24.0])

        return np.concatenate([load, generation, soc, line_loading, time_feature]).astype(np.float32)

    def _simulate_power_flow(self, line_setpoints: np.ndarray) -> Dict[str, float]:
        """Simplified power flow simulation."""
        t = self.current_step % self.load_profiles.shape[1]

        total_load = self.load_profiles[:, t].sum()
        total_generation = self.renewable_profiles[:, t].sum()
        storage_discharge = np.sum(
            np.where(self.storage_soc > 0.1,
                     self.storage_max_power * np.random.uniform(0, 0.5, self.n_storage), 0)
        )

        total_supply = total_generation + storage_discharge
        demand_satisfaction = min(total_supply / (total_load + 1e-8), 1.0)

        # Power loss increases with line loading deviation from setpoints
        line_loading = np.clip(np.abs(line_setpoints), 0, 1)
        power_loss = np.sum(line_loading ** 2) * 0.02  # I^2 * R losses (simplified)

        # Voltage stability (penalize large setpoint variations)
        voltage_deviation = np.std(line_setpoints) * 0.1

        # Renewable utilization
        renewable_utilization = min(total_generation / (total_load + 1e-8), 1.0)

        return {
            "total_loss": float(power_loss),
            "voltage_deviation": float(voltage_deviation),
            "renewable_utilization": float(renewable_utilization),
            "demand_satisfaction": float(demand_satisfaction),
        }

    def _update_storage(self, commands: np.ndarray) -> None:
        """Update battery storage state of charge based on commands."""
        for i in range(self.n_storage):
            delta_soc = commands[i] * self.storage_max_power[i] / self.storage_capacity[i]
            self.storage_soc[i] = np.clip(self.storage_soc[i] + delta_soc, 0.05, 0.95)

    def _generate_load_profiles(self) -> np.ndarray:
        """Generate synthetic hourly load profiles for all nodes."""
        hours = self.max_steps + 48
        time = np.arange(hours)

        profiles = np.zeros((self.n_nodes, hours))
        for i in range(self.n_nodes):
            base_load = np.random.uniform(20, 100)
            daily_pattern = 0.3 * np.sin(2 * np.pi * time / 24 - np.pi / 2) + 1.0
            weekly_pattern = 0.1 * np.sin(2 * np.pi * time / (24 * 7)) + 1.0
            noise = np.random.normal(0, 0.05, hours)
            profiles[i] = base_load * daily_pattern * weekly_pattern * (1 + noise)

        return np.clip(profiles, 5, None)

    def _generate_renewable_profiles(self) -> np.ndarray:
        """Generate synthetic solar/wind generation profiles."""
        hours = self.max_steps + 48
        time = np.arange(hours)

        profiles = np.zeros((self.n_nodes, hours))
        for i in range(self.n_nodes):
            if np.random.random() < 0.4:  # 40% of nodes have renewables
                capacity = np.random.uniform(10, 80)
                # Solar: peaks at noon
                solar = np.maximum(0, np.sin(2 * np.pi * time / 24 - np.pi / 2))
                # Add weather variability
                cloud_factor = np.random.uniform(0.3, 1.0, hours)
                profiles[i] = capacity * solar * cloud_factor

        return np.clip(profiles, 0, None)

    def _create_grid_topology(self) -> np.ndarray:
        """Create a simplified grid topology (adjacency matrix)."""
        adj = np.zeros((self.n_nodes, self.n_nodes))
        # Create a ring topology with some cross-connections
        for i in range(self.n_nodes):
            adj[i, (i + 1) % self.n_nodes] = 1
            adj[(i + 1) % self.n_nodes, i] = 1
        # Add a few cross-connections
        for _ in range(self.n_lines - self.n_nodes):
            i, j = np.random.choice(self.n_nodes, 2, replace=False)
            adj[i, j] = adj[j, i] = 1
        return adj

    def render(self) -> Optional[np.ndarray]:
        """Render the current grid state."""
        if self.render_mode == "human":
            t = self.current_step % self.load_profiles.shape[1]
            print(f"Step {self.current_step:4d} | "
                  f"Total Load: {self.load_profiles[:, t].sum():.1f} MW | "
                  f"Renewable: {self.renewable_profiles[:, t].sum():.1f} MW | "
                  f"Storage SoC: {self.storage_soc.mean():.2f}")
        return None

    def close(self) -> None:
        pass


# Register environment with Gymnasium
def make_smart_grid_env(**kwargs):
    """Factory function to create SmartGridEnv instances."""
    return SmartGridEnv(**kwargs)


if __name__ == "__main__":
    # Quick sanity check
    env = SmartGridEnv(n_nodes=14, n_lines=20, n_storage=3, render_mode="human")
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    total_reward = 0
    for step in range(24):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated or truncated:
            break

    print(f"\nEpisode finished after {step + 1} steps")
    print(f"Total reward: {total_reward:.3f}")
    env.close()

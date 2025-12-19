"""
CosmosEnvironment - Simulated grid world for embodied learning.

A rich environment that provides learning signal without robotics complexity.
To be fully implemented in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import random

import torch
import numpy as np


class CellType(IntEnum):
    """Types of cells in the grid."""

    EMPTY = 0
    WALL = 1
    ENERGY = 2
    HAZARD = 3
    GOAL = 4


@dataclass
class Agent:
    """An agent in the environment."""

    position: tuple[int, int]
    velocity: tuple[float, float] = (0.0, 0.0)
    energy: float = 1.0
    damage: float = 0.0
    recent_damage: float = 0.0


@dataclass
class Observation:
    """Multimodal observation for an agent."""

    # Proprioception
    position: tuple[int, int]
    velocity: tuple[float, float]
    energy: float
    damage: float

    # Exteroception
    vision: np.ndarray  # Local vision grid
    proximity: dict[str, float]  # Distance to nearest objects

    # Interoception
    hunger: float
    pain: float
    goal_distance: float

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert observation to flat tensor."""
        proprio = torch.tensor(
            [self.position[0], self.position[1], self.velocity[0], self.velocity[1],
             self.energy, self.damage],
            dtype=torch.float32,
            device=device,
        )
        vision_flat = torch.tensor(
            self.vision.flatten(), dtype=torch.float32, device=device
        )
        proximity = torch.tensor(
            [self.proximity.get("resource", 0), self.proximity.get("hazard", 0),
             self.proximity.get("agent", 0)],
            dtype=torch.float32,
            device=device,
        )
        intero = torch.tensor(
            [self.hunger, self.pain, self.goal_distance],
            dtype=torch.float32,
            device=device,
        )
        return torch.cat([proprio, vision_flat, proximity, intero])


class CosmosEnvironment:
    """
    Simulated grid world for embodied agent learning.

    Features:
    - Grid with resources, hazards, walls
    - Multiple agents (cooperation/competition)
    - Physics: movement costs, collision, decay
    - Rich observation space (proprio, extero, intero)
    """

    def __init__(
        self,
        grid_size: int = 64,
        num_resources: int = 20,
        num_hazards: int = 10,
        num_agents: int = 1,
        vision_radius: int = 5,
    ):
        self.grid_size = grid_size
        self.num_resources = num_resources
        self.num_hazards = num_hazards
        self.vision_radius = vision_radius

        # Initialize grid
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        self._place_objects()

        # Initialize agents
        self.agents: list[Agent] = []
        for _ in range(num_agents):
            pos = self._random_empty_position()
            self.agents.append(Agent(position=pos))

        # Goal position
        self.goal_position = self._random_empty_position()
        self.grid[self.goal_position] = CellType.GOAL

        self.step_count = 0

    def _random_empty_position(self) -> tuple[int, int]:
        """Find a random empty cell."""
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if self.grid[x, y] == CellType.EMPTY:
                return (x, y)

    def _place_objects(self) -> None:
        """Place resources and hazards on the grid."""
        # Place walls (border)
        self.grid[0, :] = CellType.WALL
        self.grid[-1, :] = CellType.WALL
        self.grid[:, 0] = CellType.WALL
        self.grid[:, -1] = CellType.WALL

        # Place resources
        for _ in range(self.num_resources):
            pos = self._random_empty_position()
            self.grid[pos] = CellType.ENERGY

        # Place hazards
        for _ in range(self.num_hazards):
            pos = self._random_empty_position()
            self.grid[pos] = CellType.HAZARD

    def _get_local_vision(self, position: tuple[int, int]) -> np.ndarray:
        """Get local vision grid centered on position."""
        r = self.vision_radius
        size = 2 * r + 1
        vision = np.zeros((size, size), dtype=np.int32)

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = position[0] + dx, position[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    vision[dx + r, dy + r] = self.grid[x, y]
                else:
                    vision[dx + r, dy + r] = CellType.WALL

        return vision

    def _get_proximity_sensors(self, position: tuple[int, int]) -> dict[str, float]:
        """Get distance to nearest objects of each type."""
        proximity = {"resource": float("inf"), "hazard": float("inf"), "agent": float("inf")}

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                dist = abs(x - position[0]) + abs(y - position[1])  # Manhattan
                cell = self.grid[x, y]

                if cell == CellType.ENERGY and dist < proximity["resource"]:
                    proximity["resource"] = dist
                elif cell == CellType.HAZARD and dist < proximity["hazard"]:
                    proximity["hazard"] = dist

        # Check other agents
        for agent in self.agents:
            if agent.position != position:
                dist = abs(agent.position[0] - position[0]) + abs(agent.position[1] - position[1])
                if dist < proximity["agent"]:
                    proximity["agent"] = dist

        # Normalize
        max_dist = self.grid_size * 2
        return {k: 1.0 - min(v / max_dist, 1.0) for k, v in proximity.items()}

    def get_observation(self, agent_id: int) -> Observation:
        """Get observation for an agent."""
        agent = self.agents[agent_id]

        vision = self._get_local_vision(agent.position)
        proximity = self._get_proximity_sensors(agent.position)
        goal_dist = abs(agent.position[0] - self.goal_position[0]) + \
                    abs(agent.position[1] - self.goal_position[1])

        return Observation(
            position=agent.position,
            velocity=agent.velocity,
            energy=agent.energy,
            damage=agent.damage,
            vision=vision,
            proximity=proximity,
            hunger=max(0, 0.3 - agent.energy),
            pain=agent.recent_damage,
            goal_distance=goal_dist / (self.grid_size * 2),
        )

    def step(self, actions: list[int]) -> tuple[list[Observation], list[float], list[bool]]:
        """
        Advance simulation by one timestep.

        Actions: 0=stay, 1=up, 2=down, 3=left, 4=right

        Returns:
            observations: List of observations for each agent
            rewards: List of rewards for each agent
            dones: List of done flags for each agent
        """
        self.step_count += 1

        # Action to direction mapping
        directions = {
            0: (0, 0),   # stay
            1: (-1, 0),  # up
            2: (1, 0),   # down
            3: (0, -1),  # left
            4: (0, 1),   # right
        }

        rewards = []
        dones = []

        for agent_id, action in enumerate(actions):
            agent = self.agents[agent_id]
            dx, dy = directions.get(action, (0, 0))

            # Compute new position
            new_x = max(1, min(self.grid_size - 2, agent.position[0] + dx))
            new_y = max(1, min(self.grid_size - 2, agent.position[1] + dy))

            # Check for wall collision
            if self.grid[new_x, new_y] != CellType.WALL:
                agent.position = (new_x, new_y)

            # Update velocity
            agent.velocity = (float(dx), float(dy))

            # Energy cost for movement
            if action != 0:
                agent.energy -= 0.01

            # Check cell effects
            cell = self.grid[agent.position]
            reward = 0.0
            done = False

            if cell == CellType.ENERGY:
                agent.energy = min(1.0, agent.energy + 0.2)
                reward += 0.5
                self.grid[agent.position] = CellType.EMPTY
            elif cell == CellType.HAZARD:
                agent.damage += 0.1
                agent.recent_damage = 0.1
                reward -= 0.5
            elif cell == CellType.GOAL:
                reward += 1.0
                done = True
            else:
                agent.recent_damage = 0.0

            # Death conditions
            if agent.energy <= 0 or agent.damage >= 1.0:
                done = True
                reward -= 1.0

            rewards.append(reward)
            dones.append(done)

        # Get observations
        observations = [self.get_observation(i) for i in range(len(self.agents))]

        return observations, rewards, dones

    def reset(self) -> list[Observation]:
        """Reset environment to initial state."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self._place_objects()

        for agent in self.agents:
            agent.position = self._random_empty_position()
            agent.energy = 1.0
            agent.damage = 0.0
            agent.recent_damage = 0.0
            agent.velocity = (0.0, 0.0)

        self.goal_position = self._random_empty_position()
        self.grid[self.goal_position] = CellType.GOAL

        self.step_count = 0

        return [self.get_observation(i) for i in range(len(self.agents))]

    @property
    def observation_dim(self) -> int:
        """Dimension of flattened observation."""
        vision_size = (2 * self.vision_radius + 1) ** 2
        return 6 + vision_size + 3 + 3  # proprio + vision + proximity + intero

"""
CosmosEnvironment - Simulated grid world for embodied learning.

A rich environment that provides learning signal without robotics complexity.
Phase 2: Enhanced with resource respawn, multiple resource types, and better physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional
import random

import torch
import numpy as np


class CellType(IntEnum):
    """Types of cells in the grid."""

    EMPTY = 0
    WALL = 1
    ENERGY = 2  # Basic energy resource
    HAZARD = 3
    GOAL = 4
    FOOD = 5  # Food resource (restores more energy)
    WATER = 6  # Water resource (required for survival)
    MATERIAL = 7  # Building material (for future crafting)


@dataclass
class Agent:
    """An agent in the environment."""

    position: tuple[int, int]
    velocity: tuple[float, float] = (0.0, 0.0)
    energy: float = 1.0
    damage: float = 0.0
    recent_damage: float = 0.0

    # Resource needs
    hunger: float = 0.0  # Increases over time, need food
    thirst: float = 0.0  # Increases over time, need water

    # Statistics
    resources_collected: int = 0
    cells_visited: int = 0
    steps_survived: int = 0


@dataclass
class ResourceSpawn:
    """Tracks resource respawn locations and timing."""

    position: tuple[int, int]
    resource_type: CellType
    respawn_time: int  # Step count when it respawns
    original_type: CellType  # For tracking what was there


@dataclass
class EnvironmentConfig:
    """Configuration for the environment."""

    grid_size: int = 64
    num_resources: int = 20
    num_hazards: int = 10
    num_agents: int = 1
    vision_radius: int = 5

    # Resource respawn
    enable_respawn: bool = True
    respawn_delay: int = 50  # Steps before resource respawns

    # Resource types
    num_food: int = 10
    num_water: int = 10
    num_material: int = 5

    # Survival mechanics
    hunger_rate: float = 0.005  # Hunger increase per step
    thirst_rate: float = 0.008  # Thirst increase per step
    starvation_threshold: float = 1.0  # Die if hunger/thirst exceed this

    # Movement
    movement_cost: float = 0.01
    stay_cost: float = 0.002  # Small cost even for staying still

    # Max episode length
    max_steps: int = 500


def _internal_wall_plan(grid_size: int) -> tuple[int, int]:
    """Return the segment count and maximum length used by wall generation."""
    return grid_size // 10, min(8, grid_size // 4)


def worst_case_internal_wall_cells(grid_size: int) -> int:
    """Return the maximum wall placements attempted by the maze generator.

    ``_add_internal_walls`` creates ``grid_size // 10`` segments, each with a
    maximum length of ``min(8, grid_size // 4)``.  Every placement coin flip
    can succeed and the segments can be disjoint, so their product is the
    fail-closed capacity budget.  Grids whose maximum segment is shorter than
    the generator's three-cell minimum create no internal walls.
    """
    num_segments, max_length = _internal_wall_plan(grid_size)
    if num_segments == 0 or max_length < 3:
        return 0
    return num_segments * max_length


def validate_environment_capacity(config: EnvironmentConfig) -> None:
    """Reject configurations that cannot support every placement phase.

    Reserving the worst-case internal-wall budget keeps object density from
    silently changing the maze-generation distribution and guarantees space
    for every agent plus the goal before any randomized placement begins.
    """
    integer_fields = {
        "grid_size": config.grid_size,
        "num_resources": config.num_resources,
        "num_hazards": config.num_hazards,
        "num_agents": config.num_agents,
        "num_food": config.num_food,
        "num_water": config.num_water,
        "num_material": config.num_material,
    }
    for name, value in integer_fields.items():
        if type(value) is not int:
            raise TypeError(f"{name} must be a native int")
    if config.grid_size < 4:
        raise ValueError("grid_size must be at least 4")
    if config.num_agents < 1:
        raise ValueError("num_agents must be at least 1")

    object_counts = {
        "num_resources": config.num_resources,
        "num_hazards": config.num_hazards,
        "num_food": config.num_food,
        "num_water": config.num_water,
        "num_material": config.num_material,
    }
    if any(value < 0 for value in object_counts.values()):
        raise ValueError("environment object counts must be non-negative")

    interior_cells = (config.grid_size - 2) ** 2
    objects = sum(object_counts.values())
    wall_budget = worst_case_internal_wall_cells(config.grid_size)
    reserved = config.num_agents + 1  # distinct agent cells and the goal
    required = objects + wall_budget + reserved
    if required > interior_cells:
        raise ValueError(
            "environment requires "
            f"{required} interior cells ({objects} objects, {wall_budget} "
            f"worst-case internal walls, {reserved} agent/goal reservations), "
            f"but a {config.grid_size}x{config.grid_size} grid has only "
            f"{interior_cells}"
        )


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
    thirst: float
    pain: float
    goal_distance: float

    # Statistics (for monitoring)
    steps_survived: int = 0
    resources_collected: int = 0

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
            [self.proximity.get("energy", 0), self.proximity.get("food", 0),
             self.proximity.get("water", 0), self.proximity.get("hazard", 0),
             self.proximity.get("agent", 0), self.proximity.get("goal", 0)],
            dtype=torch.float32,
            device=device,
        )
        intero = torch.tensor(
            [self.hunger, self.thirst, self.pain, self.goal_distance],
            dtype=torch.float32,
            device=device,
        )
        return torch.cat([proprio, vision_flat, proximity, intero])


class CosmosEnvironment:
    """
    Simulated grid world for embodied agent learning.

    Features:
    - Grid with multiple resource types (energy, food, water, material)
    - Resource respawn mechanics
    - Multiple agents (cooperation/competition)
    - Physics: movement costs, collision, decay
    - Survival mechanics (hunger, thirst)
    - Rich observation space (proprio, extero, intero)
    """

    def __init__(
        self,
        grid_size: int = 64,
        num_resources: int = 20,
        num_hazards: int = 10,
        num_agents: int = 1,
        vision_radius: int = 5,
        config: Optional[EnvironmentConfig] = None,
    ):
        # Use config if provided, otherwise use parameters
        if config is not None:
            self.config = config
        else:
            self.config = EnvironmentConfig(
                grid_size=grid_size,
                num_resources=num_resources,
                num_hazards=num_hazards,
                num_agents=num_agents,
                vision_radius=vision_radius,
            )

        validate_environment_capacity(self.config)

        self.grid_size = self.config.grid_size
        self.num_resources = self.config.num_resources
        self.num_hazards = self.config.num_hazards
        self.vision_radius = self.config.vision_radius

        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self._place_objects()

        # Initialize agents
        self.agents: List[Agent] = []
        occupied_positions: set[tuple[int, int]] = set()
        for _ in range(self.config.num_agents):
            pos = self._random_empty_position(excluded=occupied_positions)
            self.agents.append(Agent(position=pos))
            occupied_positions.add(pos)

        # Goal position
        self.goal_position = self._random_empty_position(
            excluded=occupied_positions
        )
        self.grid[self.goal_position] = CellType.GOAL

        # Resource respawn tracking
        self.pending_respawns: List[ResourceSpawn] = []

        # Visited cells tracking (for exploration metrics)
        self.visited_cells: set = set()

        self.step_count = 0

    def _random_empty_position(
        self,
        excluded: Optional[set[tuple[int, int]]] = None,
    ) -> tuple[int, int]:
        """Choose an empty cell or fail instead of looping indefinitely."""
        excluded = excluded or set()
        candidates = [
            (int(x), int(y))
            for x, y in np.argwhere(self.grid == CellType.EMPTY)
            if (int(x), int(y)) not in excluded
        ]
        if not candidates:
            raise RuntimeError("environment has no unreserved empty cell")
        return random.choice(candidates)

    def _place_objects(self) -> None:
        """Place resources and hazards on the grid."""
        # Place walls (border)
        self.grid[0, :] = CellType.WALL
        self.grid[-1, :] = CellType.WALL
        self.grid[:, 0] = CellType.WALL
        self.grid[:, -1] = CellType.WALL

        # Place basic energy resources
        for _ in range(self.num_resources):
            pos = self._random_empty_position()
            self.grid[pos] = CellType.ENERGY

        # Place food resources
        for _ in range(self.config.num_food):
            pos = self._random_empty_position()
            self.grid[pos] = CellType.FOOD

        # Place water resources
        for _ in range(self.config.num_water):
            pos = self._random_empty_position()
            self.grid[pos] = CellType.WATER

        # Place material resources
        for _ in range(self.config.num_material):
            pos = self._random_empty_position()
            self.grid[pos] = CellType.MATERIAL

        # Place hazards
        for _ in range(self.num_hazards):
            pos = self._random_empty_position()
            self.grid[pos] = CellType.HAZARD

        # Add some internal walls for maze-like structure
        self._add_internal_walls()

    def _add_internal_walls(self) -> None:
        """Add internal walls to create maze-like structure."""
        # Add a few random wall segments
        num_segments, max_length = _internal_wall_plan(self.grid_size)
        if num_segments == 0 or max_length < 3:
            return

        # Agent positions and the goal are selected after wall placement.
        reserved_empty_cells = self.config.num_agents + 1
        remaining_empty_cells = int(np.count_nonzero(self.grid == CellType.EMPTY))

        for _ in range(num_segments):
            # Random starting position
            x = random.randint(2, self.grid_size - 3)
            y = random.randint(2, self.grid_size - 3)

            # Random direction and length
            horizontal = random.random() < 0.5
            length = random.randint(3, max_length)

            # Place wall segment with gaps
            for i in range(length):
                if random.random() > 0.2:  # 80% chance to place wall, 20% gap
                    if horizontal:
                        wx, wy = x, min(self.grid_size - 2, y + i)
                    else:
                        wx, wy = min(self.grid_size - 2, x + i), y

                    # Only place if empty
                    if self.grid[wx, wy] == CellType.EMPTY:
                        if remaining_empty_cells <= reserved_empty_cells:
                            return
                        self.grid[wx, wy] = CellType.WALL
                        remaining_empty_cells -= 1

    def _process_respawns(self) -> None:
        """Process pending resource respawns."""
        if not self.config.enable_respawn:
            return

        still_pending = []
        for spawn in self.pending_respawns:
            if self.step_count >= spawn.respawn_time:
                # Check if position is still empty
                if self.grid[spawn.position] == CellType.EMPTY:
                    self.grid[spawn.position] = spawn.resource_type
            else:
                still_pending.append(spawn)

        self.pending_respawns = still_pending

    def _schedule_respawn(self, position: tuple[int, int], resource_type: CellType) -> None:
        """Schedule a resource to respawn."""
        if self.config.enable_respawn:
            respawn = ResourceSpawn(
                position=position,
                resource_type=resource_type,
                respawn_time=self.step_count + self.config.respawn_delay,
                original_type=resource_type,
            )
            self.pending_respawns.append(respawn)

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
        proximity = {
            "energy": float("inf"),
            "food": float("inf"),
            "water": float("inf"),
            "hazard": float("inf"),
            "agent": float("inf"),
            "goal": float("inf"),
        }

        # Map cell types to proximity keys
        cell_to_key = {
            CellType.ENERGY: "energy",
            CellType.FOOD: "food",
            CellType.WATER: "water",
            CellType.HAZARD: "hazard",
            CellType.GOAL: "goal",
        }

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                dist = abs(x - position[0]) + abs(y - position[1])  # Manhattan
                cell = self.grid[x, y]

                key = cell_to_key.get(cell)
                if key and dist < proximity[key]:
                    proximity[key] = dist

        # Check other agents
        for agent in self.agents:
            if agent.position != position:
                dist = abs(agent.position[0] - position[0]) + abs(agent.position[1] - position[1])
                if dist < proximity["agent"]:
                    proximity["agent"] = dist

        # Normalize (closer = higher value)
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
            hunger=agent.hunger,
            thirst=agent.thirst,
            pain=agent.recent_damage,
            goal_distance=goal_dist / (self.grid_size * 2),
            steps_survived=agent.steps_survived,
            resources_collected=agent.resources_collected,
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

        # Process resource respawns
        self._process_respawns()

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
        infos = []

        for agent_id, action in enumerate(actions):
            agent = self.agents[agent_id]
            dx, dy = directions.get(action, (0, 0))

            old_position = agent.position

            # Compute new position
            new_x = max(1, min(self.grid_size - 2, agent.position[0] + dx))
            new_y = max(1, min(self.grid_size - 2, agent.position[1] + dy))

            # Check for wall collision
            if self.grid[new_x, new_y] != CellType.WALL:
                agent.position = (new_x, new_y)

            # Track visited cells
            if agent.position not in self.visited_cells:
                self.visited_cells.add(agent.position)
                agent.cells_visited += 1

            # Update velocity
            agent.velocity = (float(dx), float(dy))

            # Energy cost for movement
            if action != 0:
                agent.energy -= self.config.movement_cost
            else:
                agent.energy -= self.config.stay_cost

            # Update survival needs
            agent.hunger += self.config.hunger_rate
            agent.thirst += self.config.thirst_rate
            agent.steps_survived += 1

            # Check cell effects
            cell = self.grid[agent.position]
            reward = 0.0
            done = False
            info = {"resource_collected": None}

            if cell == CellType.ENERGY:
                agent.energy = min(1.0, agent.energy + 0.2)
                reward += 0.3
                agent.resources_collected += 1
                info["resource_collected"] = "energy"
                self.grid[agent.position] = CellType.EMPTY
                self._schedule_respawn(agent.position, CellType.ENERGY)

            elif cell == CellType.FOOD:
                agent.energy = min(1.0, agent.energy + 0.3)
                agent.hunger = max(0, agent.hunger - 0.5)
                reward += 0.5
                agent.resources_collected += 1
                info["resource_collected"] = "food"
                self.grid[agent.position] = CellType.EMPTY
                self._schedule_respawn(agent.position, CellType.FOOD)

            elif cell == CellType.WATER:
                agent.thirst = max(0, agent.thirst - 0.7)
                reward += 0.4
                agent.resources_collected += 1
                info["resource_collected"] = "water"
                self.grid[agent.position] = CellType.EMPTY
                self._schedule_respawn(agent.position, CellType.WATER)

            elif cell == CellType.MATERIAL:
                reward += 0.2  # Smaller immediate reward, for future crafting
                agent.resources_collected += 1
                info["resource_collected"] = "material"
                self.grid[agent.position] = CellType.EMPTY
                self._schedule_respawn(agent.position, CellType.MATERIAL)

            elif cell == CellType.HAZARD:
                agent.damage += 0.1
                agent.recent_damage = 0.1
                reward -= 0.5

            elif cell == CellType.GOAL:
                reward += 2.0
                done = True

            else:
                agent.recent_damage = 0.0

            # Death conditions
            if agent.energy <= 0:
                done = True
                reward -= 1.0
                info["death_cause"] = "energy_depleted"

            if agent.damage >= 1.0:
                done = True
                reward -= 1.0
                info["death_cause"] = "damage"

            if agent.hunger >= self.config.starvation_threshold:
                done = True
                reward -= 0.5
                info["death_cause"] = "starvation"

            if agent.thirst >= self.config.starvation_threshold:
                done = True
                reward -= 0.5
                info["death_cause"] = "dehydration"

            # Max steps
            if self.step_count >= self.config.max_steps:
                done = True
                info["death_cause"] = "timeout"

            # Small reward for staying alive
            if not done:
                reward += 0.01

            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        # Get observations
        observations = [self.get_observation(i) for i in range(len(self.agents))]

        return observations, rewards, dones

    def reset(self) -> list[Observation]:
        """Reset environment to initial state."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self._place_objects()

        occupied_positions: set[tuple[int, int]] = set()
        for agent in self.agents:
            agent.position = self._random_empty_position(
                excluded=occupied_positions
            )
            occupied_positions.add(agent.position)
            agent.energy = 1.0
            agent.damage = 0.0
            agent.recent_damage = 0.0
            agent.velocity = (0.0, 0.0)
            agent.hunger = 0.0
            agent.thirst = 0.0
            agent.resources_collected = 0
            agent.cells_visited = 0
            agent.steps_survived = 0

        self.goal_position = self._random_empty_position(
            excluded=occupied_positions
        )
        self.grid[self.goal_position] = CellType.GOAL

        self.pending_respawns = []
        self.visited_cells = set()
        self.step_count = 0

        return [self.get_observation(i) for i in range(len(self.agents))]

    @property
    def observation_dim(self) -> int:
        """Dimension of flattened observation."""
        vision_size = (2 * self.vision_radius + 1) ** 2
        # proprio (6) + vision + proximity (6) + intero (4)
        return 6 + vision_size + 6 + 4

    def get_stats(self) -> dict:
        """Get environment statistics."""
        return {
            "step_count": self.step_count,
            "visited_cells": len(self.visited_cells),
            "coverage": len(self.visited_cells) / ((self.grid_size - 2) ** 2),
            "pending_respawns": len(self.pending_respawns),
            "agents": [
                {
                    "position": agent.position,
                    "energy": agent.energy,
                    "hunger": agent.hunger,
                    "thirst": agent.thirst,
                    "damage": agent.damage,
                    "resources_collected": agent.resources_collected,
                    "steps_survived": agent.steps_survived,
                }
                for agent in self.agents
            ],
        }

    def count_resources(self) -> dict:
        """Count remaining resources on the grid."""
        counts = {
            "energy": 0,
            "food": 0,
            "water": 0,
            "material": 0,
            "hazard": 0,
        }
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.grid[x, y]
                if cell == CellType.ENERGY:
                    counts["energy"] += 1
                elif cell == CellType.FOOD:
                    counts["food"] += 1
                elif cell == CellType.WATER:
                    counts["water"] += 1
                elif cell == CellType.MATERIAL:
                    counts["material"] += 1
                elif cell == CellType.HAZARD:
                    counts["hazard"] += 1
        return counts

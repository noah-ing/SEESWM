"""
Rendering utilities for environment visualization.

Placeholder for Phase 2 - will include:
- ASCII rendering
- Pygame rendering
- Tensorboard image logging
"""

from __future__ import annotations

import numpy as np
from .cosmos import CosmosEnvironment, CellType


def render_ascii(env: CosmosEnvironment) -> str:
    """Render environment as ASCII art."""
    chars = {
        CellType.EMPTY: ".",
        CellType.WALL: "#",
        CellType.ENERGY: "E",
        CellType.HAZARD: "X",
        CellType.GOAL: "G",
    }

    lines = []
    for x in range(env.grid_size):
        row = ""
        for y in range(env.grid_size):
            # Check if agent is here
            agent_here = False
            for i, agent in enumerate(env.agents):
                if agent.position == (x, y):
                    row += str(i)
                    agent_here = True
                    break

            if not agent_here:
                row += chars.get(env.grid[x, y], "?")

        lines.append(row)

    return "\n".join(lines)

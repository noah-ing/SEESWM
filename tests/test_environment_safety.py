"""Regression tests for bounded environment placement."""

import pytest

from src.environment.cosmos import (
    CosmosEnvironment,
    EnvironmentConfig,
    validate_environment_capacity,
    worst_case_internal_wall_cells,
)


def _sparse_config(grid_size: int) -> EnvironmentConfig:
    return EnvironmentConfig(
        grid_size=grid_size,
        num_resources=0,
        num_hazards=0,
        num_agents=1,
        vision_radius=1,
        num_food=0,
        num_water=0,
        num_material=0,
    )


@pytest.mark.parametrize("grid_size", [10, 11])
def test_small_grid_wall_generation_has_no_invalid_length_range(grid_size):
    environment = CosmosEnvironment(config=_sparse_config(grid_size))

    assert environment.agents[0].position != environment.goal_position


def test_near_capacity_grid_reserves_agent_and_goal_cells():
    # A 12x12 grid has 100 interior cells. The exact fail-closed budget is
    # three internal wall cells plus one agent and the goal, leaving 95 cells.
    config = _sparse_config(12)
    config.num_resources = 95
    environment = CosmosEnvironment(config=config)

    assert environment.agents[0].position != environment.goal_position
    observations = environment.reset()
    assert len(observations) == 1
    assert environment.agents[0].position != environment.goal_position


@pytest.mark.parametrize(
    ("grid_size", "expected"),
    [(10, 0), (11, 0), (12, 3), (32, 24), (64, 48)],
)
def test_worst_case_wall_budget_matches_generator(grid_size, expected):
    assert worst_case_internal_wall_cells(grid_size) == expected


def test_capacity_rejects_counts_that_leave_no_worst_case_wall_budget():
    config = _sparse_config(12)
    config.num_resources = 96

    with pytest.raises(ValueError, match="worst-case internal walls"):
        validate_environment_capacity(config)

    with pytest.raises(ValueError, match="worst-case internal walls"):
        CosmosEnvironment(config=config)


def test_empty_position_fails_when_no_cell_is_available():
    environment = CosmosEnvironment(config=_sparse_config(8))
    environment.grid.fill(1)

    with pytest.raises(RuntimeError, match="no unreserved empty cell"):
        environment._random_empty_position()

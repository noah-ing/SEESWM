"""
Configuration management for SEESWM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclass
class Config:
    """Main configuration container."""

    # Swarm settings
    num_agents: int = 100
    topology: str = "small_world"
    hidden_dim: int = 128
    message_passing_rounds: int = 3

    # Training settings
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    device: str = "cpu"

    # Environment settings
    grid_size: int = 64
    num_resources: int = 20
    num_hazards: int = 10

    # Neuromodulation
    dopamine_sensitivity: float = 1.0
    curiosity_weight: float = 0.5
    fear_weight: float = 0.3

    # Logging
    log_dir: str = "logs"
    save_every: int = 10
    log_every: int = 1

    # Extra settings
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "num_agents": self.num_agents,
            "topology": self.topology,
            "hidden_dim": self.hidden_dim,
            "message_passing_rounds": self.message_passing_rounds,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "device": self.device,
            "grid_size": self.grid_size,
            "num_resources": self.num_resources,
            "num_hazards": self.num_hazards,
            "dopamine_sensitivity": self.dopamine_sensitivity,
            "curiosity_weight": self.curiosity_weight,
            "fear_weight": self.fear_weight,
            "log_dir": self.log_dir,
            "save_every": self.save_every,
            "log_every": self.log_every,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        """Create from dictionary."""
        known_keys = {
            "num_agents", "topology", "hidden_dim", "message_passing_rounds",
            "learning_rate", "batch_size", "num_epochs", "device",
            "grid_size", "num_resources", "num_hazards",
            "dopamine_sensitivity", "curiosity_weight", "fear_weight",
            "log_dir", "save_every", "log_every",
        }
        kwargs = {k: v for k, v in d.items() if k in known_keys}
        extra = {k: v for k, v in d.items() if k not in known_keys}
        return cls(**kwargs, extra=extra)


def load_config(path: str | Path) -> Config:
    """Load configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return Config.from_dict(data)


def save_config(config: Config, path: str | Path) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

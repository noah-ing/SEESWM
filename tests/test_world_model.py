"""
Tests for World Model.
"""

import pytest
import torch

from src.world_model.jepa import WorldModel


class TestWorldModel:
    """Test suite for JEPA World Model."""

    def test_world_model_creation(self):
        """Test basic world model creation."""
        model = WorldModel(obs_dim=64, action_dim=4, latent_dim=128)

        assert model.obs_dim == 64
        assert model.action_dim == 4
        assert model.latent_dim == 128

    def test_forward_pass(self):
        """Test forward pass produces correct shapes."""
        model = WorldModel(obs_dim=64, action_dim=4, latent_dim=128)

        batch_size = 8
        obs = torch.randn(batch_size, 64)
        action = torch.randn(batch_size, 4)

        latent, predictions = model(obs, action)

        assert latent.shape == (batch_size, 128)
        assert len(predictions) == 3  # 3 hierarchy levels
        for pred in predictions:
            assert pred.shape == (batch_size, 128)

    def test_curiosity_computation(self):
        """Test curiosity signal computation."""
        model = WorldModel(obs_dim=32, action_dim=4, latent_dim=64)

        obs = torch.randn(4, 32)
        action = torch.randn(4, 4)
        next_obs = torch.randn(4, 32)

        curiosity = model.compute_curiosity(obs, action, next_obs)

        assert isinstance(curiosity, float)
        assert curiosity >= 0

    def test_loss_computation(self):
        """Test training loss computation."""
        model = WorldModel(obs_dim=32, action_dim=4, latent_dim=64)

        obs = torch.randn(4, 32)
        action = torch.randn(4, 4)
        next_obs = torch.randn(4, 32)

        loss = model.compute_loss(obs, action, next_obs)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_target_encoder_update(self):
        """Test EMA target encoder update."""
        model = WorldModel(obs_dim=32, action_dim=4, latent_dim=64)

        # Get initial target weights
        initial_weight = model.target_encoder.network[0].weight.clone()

        # Update main encoder
        model.encoder.network[0].weight.data += 0.1

        # Update target
        model.update_target()

        # Target should have moved slightly toward main encoder
        new_weight = model.target_encoder.network[0].weight
        assert not torch.equal(new_weight, initial_weight)

    def test_training_step(self):
        """Test a full training step."""
        model = WorldModel(obs_dim=32, action_dim=4, latent_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        obs = torch.randn(8, 32)
        action = torch.randn(8, 4)
        next_obs = torch.randn(8, 32)

        # Training step
        optimizer.zero_grad()
        loss = model.compute_loss(obs, action, next_obs)
        loss.backward()
        optimizer.step()
        model.update_target()

        # Should not error and loss should be finite
        assert torch.isfinite(loss)

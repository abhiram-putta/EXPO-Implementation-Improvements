"""Smoke test: the diffusion base policy can overfit a tiny (s, a) dataset.

If this fails, something is wrong with the noise schedule, denoising network,
or sampling loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from expo.models.diffusion_policy import DiffusionPolicy


def test_overfit_constant_target():
    torch.manual_seed(0)
    state_dim, action_dim = 4, 2
    policy = DiffusionPolicy(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=64, num_blocks=2, num_timesteps=10,
    )
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Tiny dataset: 16 states each mapped to a single fixed action [0.5, -0.5]
    states = torch.randn(16, state_dim)
    target_action = torch.tensor([[0.5, -0.5]]).expand(16, -1).clone()

    losses = []
    for step in range(2000):
        loss = policy.diffusion_loss(states, target_action)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    initial = sum(losses[:50]) / 50
    final = sum(losses[-50:]) / 50
    print(f"  initial loss ~ {initial:.4f}")
    print(f"  final   loss ~ {final:.4f}")
    assert final < initial * 0.5, f"loss did not decrease enough: {initial} → {final}"

    # Sample many times and check the average sample is near the target
    with torch.no_grad():
        samples = []
        for _ in range(8):
            samples.append(policy.sample(states))
        avg_sample = torch.stack(samples).mean(dim=0)  # [16, 2]
        sample_mean = avg_sample.mean(dim=0)           # [2]
    print(f"  mean sample after training: {sample_mean.tolist()}")
    # Loose tolerance — diffusion is stochastic, only 16 examples to fit.
    assert torch.allclose(sample_mean, torch.tensor([0.5, -0.5]), atol=0.25), \
        f"samples drifted: {sample_mean.tolist()} vs target [0.5, -0.5]"
    print("test_overfit_constant_target: OK")


if __name__ == "__main__":
    test_overfit_constant_target()
    print("\nDiffusion overfit test passed.")

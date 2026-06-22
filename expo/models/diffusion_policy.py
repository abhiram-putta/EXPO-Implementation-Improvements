from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import Mish, ResidualBlock, SinusoidalTimeEmbed


def vp_beta_schedule(num_timesteps: int, beta_min: float = 0.1,
                     beta_max: float = 20.0) -> torch.Tensor:
    """Variance-preserving β schedule derived from the VP-SDE.

    Earlier versions used `linspace(1e-4, 0.02, T)` — that is the standard
    DDPM-T=1000 linear schedule, and applying it to T=10 leaves
    `alpha_bar[T-1] ≈ 0.9` (i.e. the noisiest training input still contains
    ~95% of the signal, while the sampler initializes from pure N(0, I) —
    a distribution shift the denoiser never trained on).

    The proper VP-SDE-derived schedule keeps `alpha_bar[T-1] ≈ 0` for any T.
    For VP-SDE β(t) = β_min + t·(β_max − β_min) we have
        alpha_bar(t) = exp(-(β_min·t + 0.5·(β_max − β_min)·t²))
    Discretize at t_k = k/T and recover discrete betas as 1 − α_t / α_{t-1}.

    Defaults β_min=0.1, β_max=20.0 are the canonical VP-SDE values
    (Song et al. 2020 — score-based generative modeling through SDEs).
    """
    t = torch.arange(num_timesteps + 1).float() / num_timesteps
    log_alpha_bars = -(beta_min * t + 0.5 * (beta_max - beta_min) * t.pow(2))
    alpha_bars_full = log_alpha_bars.exp()
    alphas = alpha_bars_full[1:] / alpha_bars_full[:-1]
    return 1.0 - alphas


class DenoiseNet(nn.Module):
    """Residual MLP that predicts the noise ε at timestep t given (x_t, s).

    Architecture follows Implementation Guide §4.1: state + time embed +
    noisy action concatenated into a hidden_dim trunk; `num_blocks` residual
    blocks; linear output head.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_blocks: int = 3, time_embed_dim: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            Mish(),
        )
        self.input_proj = nn.Linear(state_dim + action_dim + hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, x_t: torch.Tensor, state: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = torch.cat([x_t, state, t_emb], dim=-1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class DiffusionPolicy(nn.Module):
    """DDPM-based base policy with VP schedule.

    Two roles:
      - `diffusion_loss(state, action)`: ε-prediction MSE for IL training.
      - `sample(state)`: ancestral DDPM sampling, used by the OTF extractor.

    The output of `sample` is clipped to [-1, 1] (the agent-side action range).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_blocks: int = 3, num_timesteps: int = 10):
        super().__init__()
        self.action_dim = action_dim
        self.num_timesteps = num_timesteps

        self.net = DenoiseNet(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dim=hidden_dim, num_blocks=num_blocks,
        )

        betas = vp_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def diffusion_loss(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B = action.shape[0]
        device = action.device
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        noise = torch.randn_like(action)
        sqrt_ab = self.sqrt_alpha_bars[t].unsqueeze(-1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        x_t = sqrt_ab * action + sqrt_one_minus_ab * noise
        pred_noise = self.net(x_t, state, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, state: torch.Tensor) -> torch.Tensor:
        """Ancestral DDPM sampling (Implementation Guide §9.2)."""
        B = state.shape[0]
        device = state.device
        x = torch.randn(B, self.action_dim, device=device)
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self.net(x, state, t_tensor)
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
            )
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean
        return x.clamp(-1.0, 1.0)

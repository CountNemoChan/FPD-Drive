"""DDIM trajectory refinement using the released student weights."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)



def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)



class _RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE head dimension must be even.")
        self.register_buffer("inv_freq", 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim)), persistent=False)
        self.max_positions = int(max_positions)

    def forward(
        self,
        x: torch.Tensor,
        length: int,
        *,
        offset: int = 0,
        positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions is None:
            pos = torch.arange(
                int(offset), int(offset) + length,
                device=x.device, dtype=self.inv_freq.dtype,
            )
        else:
            if positions.ndim != 1 or positions.numel() != length:
                raise ValueError(
                    "Temporal RoPE positions must be a 1D tensor matching the "
                    f"token length: got {tuple(positions.shape)} for length={length}."
                )
            pos = positions.to(device=x.device, dtype=self.inv_freq.dtype)
        phase = torch.einsum("i,j->ij", pos, self.inv_freq.to(device=x.device))
        phase = torch.cat((phase, phase), dim=-1)
        return phase.cos()[None, None].to(dtype=x.dtype), phase.sin()[None, None].to(dtype=x.dtype)



class _DiTAttention(nn.Module):
    """Refiner attention: RMS-normalized q/k and RoPE for self attention."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("DiT dim must be divisible by heads.")
        self.heads, self.head_dim = int(heads), dim // heads
        self.to_q, self.to_k, self.to_v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.q_norm, self.k_norm = _RMSNorm(self.head_dim), _RMSNorm(self.head_dim)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rotary: Optional[_RotaryEmbedding] = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        b, nq, _ = x.shape
        nk = context.shape[1]
        q = self.to_q(x).view(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(b, nk, self.heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if rotary is not None:
            cos, sin = rotary(x, nq)
            q = q * cos + _rotate_half(q) * sin
            if context is x:
                k = k * cos + _rotate_half(k) * sin

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=0.0,
        )
        return self.out(y.transpose(1, 2).reshape(b, nq, -1))



class _DiTBlock(nn.Module):
    """adaLN-Zero self-attention, scene cross-attention, and FFN."""

    def __init__(self, dim: int, heads: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = _DiTAttention(dim, heads, dropout)
        self.cross_attn = _DiTAttention(dim, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    @staticmethod
    def _mod(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor,
                rotary: _RotaryEmbedding) -> torch.Tensor:
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = self.ada(cond).chunk(9, dim=-1)
        self_update = self.self_attn(self._mod(self.norm1(x), s1, c1), rotary=rotary)
        x = x + g1[:, None] * self_update
        cross_update = self.cross_attn(self._mod(self.norm2(x), s2, c2), context=context)
        x = x + g2[:, None] * cross_update
        x = x + g3[:, None] * self.ffn(self._mod(self.norm3(x), s3, c3))
        return x


class Refiner(nn.Module):
    """Refine proposals with deterministic DDIM conditioned on scene and ego tokens."""

    def __init__(
        self, *, dim: int, num_poses: int, state_size: int = 3,
        num_layers: int = 12, num_heads: int = 8, hidden_dim: int = 1024,
        dropout: float = 0.0, num_inference_steps: int = 12, ego_status_dim: int = 8,
        diffusion_type: str = "ddim", ddim_num_train_timesteps: int = 1000,
        ddim_beta_schedule: str = "scaled_linear", ddim_beta_start: float = 0.0001,
        ddim_beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 2:
            raise ValueError("Refiner requires even per-head dimension.")
        self.num_poses, self.state_size = int(num_poses), int(state_size)
        self.num_inference_steps = int(num_inference_steps)
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        self.diffusion_type = str(diffusion_type).strip().lower()
        if self.diffusion_type != "ddim":
            raise ValueError("The inference Refiner requires diffusion_type=ddim.")
        self.ddim_num_train_timesteps = int(ddim_num_train_timesteps)
        if self.ddim_num_train_timesteps <= 0:
            raise ValueError("ddim_num_train_timesteps must be positive.")
        self.ddim_beta_schedule = str(ddim_beta_schedule).strip().lower()
        if self.ddim_beta_schedule == "scaled_linear":
            betas = torch.linspace(
                math.sqrt(float(ddim_beta_start)), math.sqrt(float(ddim_beta_end)),
                self.ddim_num_train_timesteps, dtype=torch.float32,
            ).square()
        elif self.ddim_beta_schedule == "linear":
            betas = torch.linspace(
                float(ddim_beta_start), float(ddim_beta_end),
                self.ddim_num_train_timesteps, dtype=torch.float32,
            )
        else:
            raise ValueError("ddim_beta_schedule must be scaled_linear or linear.")
        # The diffusion schedule is configuration, not learned checkpoint state.
        self.register_buffer(
            "ddim_alphas_cumprod", torch.cumprod(1.0 - betas, dim=0), persistent=False,
        )
        self.action_proj = nn.Linear(state_size, dim)
        self.action_time = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_embed = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ego_encoder = nn.Sequential(nn.Linear(ego_status_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
        self.position = nn.Embedding(self.num_poses, dim)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        self.rotary = _RotaryEmbedding(dim // num_heads, self.num_poses)
        self.blocks = nn.ModuleList([
            _DiTBlock(dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        self.velocity_head = nn.Linear(dim, state_size)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)


    @staticmethod
    def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1))
        embedding = torch.cat(((t.float()[:, None] * freqs[None]).sin(), (t.float()[:, None] * freqs[None]).cos()), dim=-1)
        return F.pad(embedding, (0, dim - embedding.shape[-1])) if embedding.shape[-1] < dim else embedding


    def velocity(self, noisy: torch.Tensor, scene_tokens: torch.Tensor,
                 ego_status: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        b, modes, poses, _ = noisy.shape
        if poses != self.num_poses or scene_tokens.ndim != 3:
            raise ValueError("Unexpected trajectory or scene-token shape.")
        dim = scene_tokens.shape[-1]
        cond = self.time_embed(self._sinusoidal(timesteps, dim).to(dtype=scene_tokens.dtype))
        cond = cond + self.ego_encoder(ego_status.to(device=cond.device, dtype=cond.dtype))
        cond = cond[:, None].expand(-1, modes, -1).reshape(b * modes, -1)
        context = scene_tokens[:, None].expand(
            -1, modes, -1, -1
        ).reshape(b * modes, scene_tokens.shape[1], dim)
        tflat = timesteps[:, None].expand(-1, modes).reshape(-1)
        x = noisy.reshape(b * modes, poses, self.state_size)
        action = self.action_proj(x.to(dtype=self.action_proj.weight.dtype))
        time = self._sinusoidal(tflat, dim).to(dtype=action.dtype)[:, None].expand(-1, poses, -1)
        x = self.action_time(torch.cat((action, time), dim=-1)) + self.position.weight[:poses].to(dtype=action.dtype)[None]
        for block in self.blocks:
            x = block(x, context.to(dtype=x.dtype), cond.to(dtype=x.dtype), self.rotary)
        shift, scale = self.final_ada(cond.to(dtype=x.dtype)).chunk(2, dim=-1)
        x = self.final_norm(x) * (1.0 + scale[:, None]) + shift[:, None]
        velocity = self.velocity_head(x).reshape(b, modes, poses, self.state_size)
        return velocity

    def _ddim_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        if num_inference_steps > self.ddim_num_train_timesteps:
            raise ValueError("DDIM inference steps cannot exceed ddim_num_train_timesteps.")
        step_ratio = self.ddim_num_train_timesteps // num_inference_steps
        return torch.arange(
            0, num_inference_steps, device=device, dtype=torch.long
        ).mul(step_ratio).flip(0)


    def _sample_ddim(
        self, residual: torch.Tensor, base_proposals: torch.Tensor,
        scene_tokens: torch.Tensor, ego_status: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        step_ratio = self.ddim_num_train_timesteps // num_inference_steps
        for timestep in self._ddim_timesteps(num_inference_steps, residual.device):
            timestep_id = int(timestep.item())
            timesteps = torch.full(
                (base_proposals.shape[0],), timestep_id,
                device=base_proposals.device, dtype=torch.long,
            )
            pred_noise = self.velocity(residual, scene_tokens, ego_status, timesteps)
            alpha_t = self.ddim_alphas_cumprod[timestep_id].to(
                device=residual.device, dtype=residual.dtype
            )
            prev_timestep = timestep_id - step_ratio
            alpha_prev = (
                self.ddim_alphas_cumprod[prev_timestep].to(
                    device=residual.device, dtype=residual.dtype
                )
                if prev_timestep >= 0 else residual.new_tensor(1.0)
            )
            pred_delta = (
                residual - (1.0 - alpha_t).sqrt() * pred_noise
            ) / alpha_t.sqrt().clamp_min(1e-12)
            residual = (
                alpha_prev.sqrt() * pred_delta
                + (1.0 - alpha_prev).sqrt() * pred_noise
            )
        return base_proposals + residual


    @torch.no_grad()
    def sample(
        self,
        base_proposals: torch.Tensor,
        scene_tokens: torch.Tensor,
        ego_status: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        initial_residual: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample proposals, optionally using deterministic per-scene initial noise."""
        if initial_residual is None:
            residual = torch.randn(
                base_proposals.shape,
                device=base_proposals.device,
                dtype=base_proposals.dtype,
                generator=generator,
            )
        else:
            if initial_residual.shape != base_proposals.shape:
                raise ValueError(
                    "initial_residual must have the same shape as base_proposals: "
                    f"{tuple(initial_residual.shape)} != {tuple(base_proposals.shape)}"
                )
            residual = initial_residual.to(device=base_proposals.device, dtype=base_proposals.dtype)
        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        if steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        return self._sample_ddim(
            residual, base_proposals, scene_tokens, ego_status, steps,
        )

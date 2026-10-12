"""Shared student model: FPDplanner, DDIM Refiner, and checkpoint-native scorer."""
from __future__ import annotations

import hashlib
import torch
from omegaconf import DictConfig

from diffsynth.planning.fpd_planner.model import FPDplanner
from diffsynth.planning.fpd_planner.refiner import Refiner
from diffsynth.planning.fpd_planner.trajectory_selection import validate_ensemble_weights
from .navsim import _load_checkpoint_state_dict
from .selection import select_navhard_trajectory, select_stagewise, select_adaptive_top8, stage_token_map


def _scenario_generator(device, token, base_seed):
    digest = hashlib.md5(f"{token}:{base_seed}".encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _stack_scenario_noise(reference, tokens, base_seed):
    if reference.shape[0] != len(tokens):
        raise ValueError(f"Noise batch={reference.shape[0]} does not match tokens={len(tokens)}.")
    return torch.cat([
        torch.randn((1, *reference.shape[1:]), device=reference.device, dtype=reference.dtype,
                    generator=_scenario_generator(reference.device, token, base_seed))
        for token in tokens
    ], dim=0)


class FPDDriveModel(torch.nn.Module):
    """All four presets share the same parameter names and generation path."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        if cfg.model.type != 'fpdplanner':
            raise ValueError('The release requires model.type=fpdplanner')
        refiner = cfg.joint.refiner
        if not refiner.enable or refiner.type != 'ddim' or refiner.architecture != 'refiner':
            raise ValueError('The release requires the enabled DDIM Refiner')
        if refiner.candidate_mode != 'refined_only' or int(cfg.model.config.proposal_num) != 64:
            raise ValueError('The release requires exactly 64 refined candidates')
        if int(refiner.eval_noise_seed) != 27 or refiner.get('eval_noise_seeds') is not None:
            raise ValueError('The release requires the single token noise seed 27')
        self.fpd_planner = FPDplanner(cfg.model.config)
        self.refiner = Refiner(
            dim=int(cfg.model.config.tf_d_model), num_poses=int(cfg.model.config.num_poses),
            state_size=3, num_layers=int(refiner.num_layers), num_heads=int(refiner.num_heads),
            hidden_dim=int(refiner.hidden_dim), dropout=float(refiner.dropout),
            diffusion_type='ddim', num_inference_steps=int(refiner.num_inference_steps),
            ddim_num_train_timesteps=int(refiner.ddim_num_train_timesteps),
            ddim_beta_schedule=str(refiner.ddim_beta_schedule),
            ddim_beta_start=float(refiner.ddim_beta_start), ddim_beta_end=float(refiner.ddim_beta_end),
            ego_status_dim=int(refiner.ego_status_dim),
        )
        self.evaluation = cfg.evaluation
        self.ensemble_weights = validate_ensemble_weights(cfg.evaluation.get('trajectory_ensemble_weights'))
        self.stages = stage_token_map(cfg.evaluation.navsim_root) if cfg.evaluation.source == 'navhard_scene_loader' else None
        self.requires_grad_(False)
        self.eval()

    def _score_proposals(self, features, scene_features, proposals, stage_ids):
        status = features['ego_status']
        status = status.flatten(-2) if self.fpd_planner._config.full_history_status else status[:, -1]
        ego = self.fpd_planner.hist_encoding(status)[:, None]
        batch, count = proposals.shape[:2]
        embedded = self.fpd_planner.pos_embed(proposals.reshape(batch, count, -1).detach())
        tokens = self.fpd_planner.scorer_attention(embedded, scene_features) + ego
        logits, *_ = self.fpd_planner.scorer(proposals, tokens)
        if self.evaluation.get('selection_method') == 'adaptive_top8':
            return select_adaptive_top8(logits, proposals, stage_ids, self.evaluation.stage_ranking_weights)
        if self.stages is not None:
            ids = torch.as_tensor(stage_ids, device=proposals.device)
            return select_stagewise(logits, proposals, ids, self.evaluation.stage_ranking_weights,
                                    self.evaluation.stage_ranking_precision, self.ensemble_weights)
        return select_navhard_trajectory(logits, proposals, self.evaluation.ranking_weights)

    @torch.inference_mode()
    def forward(self, features, scenario_tokens):
        scene_features = self.fpd_planner.encode_scene_features(features)
        predictions = self.fpd_planner(features, scene_features=scene_features)
        base_proposals = predictions['proposals']
        ego_status = features['ego_status'][:, -1, 3:].to(device=base_proposals.device, dtype=base_proposals.dtype)
        refiner_scene_features = scene_features.to(device=base_proposals.device, dtype=base_proposals.dtype)
        noise = _stack_scenario_noise(base_proposals, scenario_tokens, 27)
        refined = self.refiner.sample(base_proposals, refiner_scene_features, ego_status, initial_residual=noise)
        # Preserve the original single-seed candidate tensor layout.
        refined = torch.cat([refined], dim=1)
        stage_ids = [self.stages[str(token)] for token in scenario_tokens] if self.stages is not None else None
        return self._score_proposals(features, scene_features, refined, stage_ids)


def load_checkpoint(model, checkpoint_path):
    state, container = _load_checkpoint_state_dict(checkpoint_path)
    state = {str(key).removeprefix('module.'): value for key, value in state.items()}
    if any(key.startswith('scorer_future_predictor.') or '.future_cross_attn.' in key for key in state):
        raise ValueError('This inference package uses the checkpoint-native student scorer without future memory')
    if 'candidate_densification_signature' in state:
        raise ValueError('The release uses exactly 64 candidates without densification')
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise RuntimeError(f'Checkpoint is missing required inference weights: {result.missing_keys[:10]}')
    print(f'[FPD-Drive] checkpoint={checkpoint_path} container={container} '
          f'missing={len(result.missing_keys)} ignored={len(result.unexpected_keys)}', flush=True)

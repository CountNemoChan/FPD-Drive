"""Candidate ranking and pose selection for the four released inference presets.

Arithmetic and stable tie ordering are preserved from the evaluated policies.
"""
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
from diffsynth.planning.fpd_planner.reweighted_ranking import reweighted_selection
from diffsynth.planning.fpd_planner.trajectory_selection import select_ranked_trajectory

PRECISION = 'bf16_sigmoid_numpy_fp64'
WEIGHT_KEYS = ('no_at_fault_collisions', 'drivable_area_compliance',
               'driving_direction_compliance', 'time_to_collision_within_bound',
               'ego_progress', 'comfort')
CFG_WEIGHT_KEYS = ('noc', 'dac', 'ddc', 'ttc', 'ep', 'comfort')

@torch.no_grad()
def select_navhard_trajectory(logits, proposals, weights, ensemble_weights=None):
    if float(weights.ddc) == 0:
        scored = reweighted_selection({'pred_logit': logits}, proposals, weights)
        if ensemble_weights is None:
            return scored['trajectory']
        return select_ranked_trajectory(proposals, scored['pdm_score'], ensemble_weights)
    # Same probability precision as the historical policy, now including DDC.
    keys = ('no_at_fault_collisions', 'drivable_area_compliance',
            'driving_direction_compliance', 'time_to_collision_within_bound',
            'ego_progress', 'comfort')
    p = torch.stack([logits[key] for key in keys], -1)
    p = p.to(device='cpu', dtype=torch.bfloat16).sigmoid().float().numpy().astype(np.float64)
    p = np.clip(p, 1e-12, 1.)
    score = (float(weights.noc) * np.log(p[..., 0])
             + float(weights.dac) * np.log(p[..., 1])
             + float(weights.ddc) * np.log(p[..., 2])
             + np.log(float(weights.ttc) * p[..., 3]
                      + float(weights.ep) * p[..., 4]
                      + float(weights.comfort) * p[..., 5]))
    return select_ranked_trajectory(
        proposals, torch.as_tensor(score, device=proposals.device), ensemble_weights)


@torch.no_grad()
def select_stagewise(logits,proposals,stage_ids,weights,precisions,ensemble_weights=None):
    # Keep blended poses in FP32 instead of rounding them back to BF16.
    output=torch.empty_like(proposals[:,0],dtype=torch.float32 if ensemble_weights is not None else proposals.dtype)
    for i,name in enumerate(('s1','s2')):
        mask=stage_ids==i
        if not mask.any():continue
        sub={k:v[mask] for k,v in logits.items()};p=proposals[mask];w=weights[name]
        if precisions[name]==PRECISION:
            chosen=select_navhard_trajectory(sub,p,w,ensemble_weights)
        else:raise ValueError(f'Unsupported ranking precision: {precisions[name]}')
        output[mask]=chosen
    return output


def stage_token_map(navsim_root):
    split=OmegaConf.load(Path(navsim_root)/'navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml')
    result={}
    def assign(token,value):
        token=str(token)
        if token in result and result[token]!=value:
            raise ValueError(f'Conflicting official stage metadata for {token}')
        result[token]=value
    for a,b,pairs in split.reactive_all_mapping:
        assign(a,0);assign(b,0)
        for x,y in pairs:
            assign(x,1);assign(y,1)
    return result


def _scores(logits_fp32, weights, precision):
    """Return [N,64] score array for one stage's weights.

    precision:
      'bf16_sigmoid_numpy_fp64' -- baseline: sigmoid computed in bf16.
      'fp32_sigmoid_numpy_fp64' -- sigmoid computed in fp32 (from bf16 logits).
    """
    if precision == "bf16_sigmoid_numpy_fp64":
        p = torch.as_tensor(logits_fp32, dtype=torch.bfloat16)
        p = p.sigmoid().float().numpy().astype(np.float64)
    elif precision == "fp32_sigmoid_numpy_fp64":
        p = torch.as_tensor(logits_fp32, dtype=torch.float32).sigmoid().numpy().astype(np.float64)
    else:
        raise ValueError(precision)
    p = np.clip(p, 1e-12, 1.0)
    score = (weights[0] * np.log(p[..., 0]) + weights[1] * np.log(p[..., 1])
             + weights[2] * np.log(p[..., 2])
             + np.log(weights[3] * p[..., 3] + weights[4] * p[..., 4]
                      + weights[5] * p[..., 5]))
    return score



def _blend(trajs, idxs, weights):
    """trajs [N,8,3]; idxs candidate indices; weights convex.

    Must reproduce trajectory_selection.select_ranked_trajectory EXACTLY:
    selected.float() * new_tensor(weights) summed in torch fp32.  bf16->fp32
    conversion is exact, so cached fp32 poses are bitwise what the evaluator
    blends.
    """
    sel = torch.as_tensor(trajs[idxs], dtype=torch.float32)  # [k,8,3]
    w = sel.new_tensor(weights)[:, None, None]
    return (sel * w).sum(dim=0).numpy()



def _ranked_indices(score, k):
    return np.argsort(-score, kind="stable")[:k]



def _softmax_standardized(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float64)
    if x.std() > 0:
        z = (x - x.mean()) / x.std()
    else:
        z = x - x.mean()
    z = z / temperature
    e = np.exp(z - z.max())
    return e / e.sum()


def select_adaptive_top8(logits, proposals, stage_ids, stage_weights):
    """Apply the original Pro V2 replay math directly to fresh model outputs."""
    logits_cpu = torch.stack([logits[key] for key in WEIGHT_KEYS], -1).detach().float().cpu().numpy().astype(np.float32)
    poses_cpu = proposals.detach().float().cpu().numpy().astype(np.float32)
    selected = []
    for i, stage in enumerate(stage_ids):
        weights = [float(stage_weights['s1' if int(stage) == 0 else 's2'][key]) for key in CFG_WEIGHT_KEYS]
        scores = _scores(logits_cpu[i], weights, 'fp32_sigmoid_numpy_fp64')
        indices = _ranked_indices(scores, 8)
        blend_weights = _softmax_standardized(scores[indices], temperature=1.0)
        selected.append(_blend(poses_cpu[i], indices, blend_weights))
    return torch.as_tensor(np.stack(selected), device=proposals.device, dtype=torch.float32)


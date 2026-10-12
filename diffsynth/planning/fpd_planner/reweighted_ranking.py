"""Inference ranking: BF16 sigmoid, historical NumPy FP64 aggregate."""
import numpy as np
import torch

RANKING_PRECISION = "bf16_sigmoid_numpy_fp64"

@torch.no_grad()
def reweighted_selection(scored, proposals, cfg):
    """Preserve the evaluated ranking arithmetic and stable tie order."""
    keys = ('no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
            'time_to_collision_within_bound', 'comfort')
    if float(cfg.ddc) != 0:
        raise ValueError('Historical reweighting requires DDC weight zero')
    # The historical experiment rounded sigmoid probabilities to BF16, then
    # used NumPy FP64 logs/sums for ranking. Preserve both precision choices.
    probabilities = torch.stack([scored['pred_logit'][k] for k in keys], -1)
    p = probabilities.to(device='cpu', dtype=torch.bfloat16).sigmoid().float().numpy().astype(np.float64)
    p = np.clip(p, 1e-12, 1.)
    score = (float(cfg.noc)*np.log(p[..., 0]) + float(cfg.dac)*np.log(p[..., 1])
             + np.log(float(cfg.ep)*p[..., 2] + float(cfg.ttc)*p[..., 3]
                      + float(cfg.comfort)*p[..., 4]))
    selected = torch.as_tensor(score.argmax(1), device=proposals.device)
    rows = torch.arange(len(proposals), device=proposals.device)
    return {**scored, 'selected_idx': selected,
            'pdm_score': torch.as_tensor(score, device=proposals.device),
            'trajectory': proposals[rows, selected]}



"""Final trajectory selection after scorer ranking (evaluation only)."""
import math

import torch


def validate_ensemble_weights(weights):
    if weights is None:
        return None
    weights = tuple(float(w) for w in weights)
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('trajectory_ensemble_weights must be finite, nonnegative and nonempty')
    if not math.isclose(sum(weights), 1., rel_tol=0., abs_tol=1e-6):
        raise ValueError('trajectory_ensemble_weights must sum to 1')
    return weights


def select_ranked_trajectory(proposals, scores, weights=None):
    """Blend top-ranked poses linearly, including heading; default is legacy top1."""
    batch = torch.arange(proposals.shape[0], device=proposals.device)
    if weights is None:
        return proposals[batch, scores.argmax(dim=1)]
    if len(weights) > proposals.shape[1]:
        raise ValueError('Not enough candidates for trajectory_ensemble_weights')
    # Stable ties retain candidate order, consistent with argmax's first maximum.
    indices = scores.argsort(dim=1, descending=True, stable=True)[:, :len(weights)]
    selected = proposals[batch[:, None], indices].float()
    coefficients = selected.new_tensor(weights)[None, :, None, None]
    return (selected * coefficients).sum(dim=1)

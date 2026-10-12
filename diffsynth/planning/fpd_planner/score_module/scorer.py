"""Checkpoint-native heads for scoring trajectory candidates."""
import torch.nn as nn


class Scorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if any(config.get(name, False) for name in
               ('double_score', 'agent_pred', 'area_pred', 'bev_map', 'bev_agent')):
            raise ValueError("The inference scorer supports only the six released score heads.")
        self.pred_score = nn.ModuleDict({
            'no_at_fault_collisions': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'drivable_area_compliance':
                nn.Sequential(
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, 1),
                ),
            'time_to_collision_within_bound': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'ego_progress': nn.Sequential(
                nn.Linear(config.tf_d_model,config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'driving_direction_compliance': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'comfort': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1)
            )
        })

    def forward(self, proposals, bev_feature):
        pred_logit = {key: head(bev_feature).squeeze(-1)
                      for key, head in self.pred_score.items()}
        return pred_logit, None, None, None, None, None, None

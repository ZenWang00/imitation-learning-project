from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class StateDiscriminator(nn.Module):
    """Binary discriminator D(s) trained to score expert states higher than replay states.

    The SOAR bonus is the log-odds: log D(s) - log(1 - D(s)).
    This encourages the policy to visit states similar to the expert's.
    """

    def __init__(self, observation_dim: int, hidden_dims: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = observation_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, observations: Tensor) -> Tensor:
        return torch.sigmoid(self.model(observations).squeeze(-1))

    def log_odds(self, observations: Tensor) -> Tensor:
        logits = self.model(observations).squeeze(-1)
        return logits  # log D/(1-D) = logit before sigmoid


def soar_discriminator_loss(
    expert_observations: Tensor,
    replay_observations: Tensor,
    discriminator: StateDiscriminator,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Binary cross-entropy loss to train the state discriminator.

    Expert states are labelled 1, replay states labelled 0.
    """
    expert_logits = discriminator.model(expert_observations).squeeze(-1)
    replay_logits = discriminator.model(replay_observations).squeeze(-1)

    expert_loss = F.binary_cross_entropy_with_logits(
        expert_logits, torch.ones_like(expert_logits)
    )
    replay_loss = F.binary_cross_entropy_with_logits(
        replay_logits, torch.zeros_like(replay_logits)
    )
    loss = expert_loss + replay_loss

    metrics = {
        "soar_disc_loss": loss.detach(),
        "expert_disc_mean": torch.sigmoid(expert_logits).detach().mean(),
        "replay_disc_mean": torch.sigmoid(replay_logits).detach().mean(),
    }
    return loss, metrics

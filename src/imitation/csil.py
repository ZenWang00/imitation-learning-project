from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RewardNetwork(nn.Module):
    """MLP reward network r_θ(s, a) → scalar."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).squeeze(-1)


def csil_reward_loss(
    expert_features: Tensor,
    replay_features: Tensor,
    reward_network: RewardNetwork,
    *,
    regularization_weight: float = 0.25,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Chi-squared divergence reward learning loss.

    Minimises -E_expert[r] + E_replay[r] + (λ/4)*E_replay[r²],
    which corresponds to the chi-squared f-divergence between
    expert and replay state-action occupancies.

    Args:
        expert_features: Concatenated (obs, action) for expert transitions.
        replay_features: Concatenated (obs, action) for replay transitions.
        reward_network: The learnable reward model.
        regularization_weight: Weight λ on the chi-squared penalty term.

    Returns:
        (loss, metrics dict)
    """
    expert_rewards = reward_network(expert_features)
    replay_rewards = reward_network(replay_features)

    expert_term = -expert_rewards.mean()
    replay_term = replay_rewards.mean()
    regularizer = regularization_weight * replay_rewards.pow(2).mean()

    loss = expert_term + replay_term + regularizer

    metrics = {
        "reward_loss": loss.detach(),
        "expert_reward_mean": expert_rewards.detach().mean(),
        "replay_reward_mean": replay_rewards.detach().mean(),
        "regularizer": regularizer.detach(),
    }
    return loss, metrics


def csil_discrete_q_loss(
    observations: Tensor,
    actions: Tensor,
    next_observations: Tensor,
    dones: Tensor,
    rewards: Tensor,
    q_network: nn.Module,
    target_q_network: nn.Module,
    *,
    gamma: float = 0.99,
    temperature: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Soft Q-learning update using the CSIL reward signal (discrete actions)."""
    with torch.no_grad():
        next_q = target_q_network(next_observations)
        next_values = temperature * torch.logsumexp(next_q / temperature, dim=-1)
        targets = rewards + gamma * (1.0 - dones) * next_values

    current_q = q_network(observations).gather(1, actions.long().unsqueeze(1)).squeeze(1)
    loss = F.mse_loss(current_q, targets)

    metrics = {
        "q_loss": loss.detach(),
        "q_mean": current_q.detach().mean(),
        "target_mean": targets.detach().mean(),
    }
    return loss, metrics


def csil_continuous_critic_loss(
    observations: Tensor,
    actions: Tensor,
    next_observations: Tensor,
    dones: Tensor,
    rewards: Tensor,
    q_network: nn.Module,
    target_q_network: nn.Module,
    actor: nn.Module,
    *,
    gamma: float = 0.99,
    temperature: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """SAC-style critic update using the CSIL reward signal (continuous actions)."""
    with torch.no_grad():
        next_actions, next_log_probs = actor.sample(next_observations)
        next_q = target_q_network(next_observations, next_actions)
        next_values = next_q - temperature * next_log_probs
        targets = rewards + gamma * (1.0 - dones) * next_values

    current_q = q_network(observations, actions)
    loss = F.mse_loss(current_q, targets)

    metrics = {
        "q_loss": loss.detach(),
        "q_mean": current_q.detach().mean(),
        "target_mean": targets.detach().mean(),
    }
    return loss, metrics


def csil_actor_loss(
    observations: Tensor,
    q_network: nn.Module,
    actor: nn.Module,
    *,
    temperature: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """SAC-style actor update (continuous actions)."""
    sampled_actions, log_probs = actor.sample(observations)
    q_values = q_network(observations, sampled_actions)
    loss = (temperature * log_probs - q_values).mean()

    metrics = {
        "actor_loss": loss.detach(),
        "entropy": (-log_probs).detach().mean(),
    }
    return loss, metrics

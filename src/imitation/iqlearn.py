from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class IQBatch:
    observations: Tensor
    actions: Tensor
    next_observations: Tensor
    dones: Tensor


class DiscreteQNetwork(nn.Module):
    """Small MLP critic for classic-control environments with discrete actions."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        input_dim = observation_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, observations: Tensor) -> Tensor:
        return self.model(observations)


def soft_value_from_q(q_values: Tensor, temperature: float = 1.0) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return temperature * torch.logsumexp(q_values / temperature, dim=-1)


def inverse_soft_q_reward(
    q_values: Tensor,
    actions: Tensor,
    next_q_values: Tensor,
    dones: Tensor,
    gamma: float,
    temperature: float = 1.0,
) -> Tensor:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    chosen_q = q_values.gather(dim=1, index=actions.long().unsqueeze(1)).squeeze(1)
    next_values = soft_value_from_q(next_q_values, temperature=temperature)
    not_done = 1.0 - dones.float()
    return chosen_q - gamma * not_done * next_values


def chi2_regularizer(rewards: Tensor) -> Tensor:
    return 0.25 * rewards.pow(2).mean()


def iq_critic_loss(
    expert_batch: IQBatch,
    replay_batch: IQBatch,
    q_network: nn.Module,
    target_q_network: nn.Module,
    *,
    gamma: float = 0.99,
    temperature: float = 1.0,
    regularization_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Compute the online IQ-Learn critic objective for expert and replay transitions.

    This follows the inverse soft-Q reward parameterization:
    r_Q(s, a) = Q(s, a) - gamma * V(s')

    The minimization loss is the negative expert reward term, a replay value
    difference term for occupancy stabilization, and the chi-square reward
    regularizer used in practical IQ-Learn implementations.
    """

    q_values = q_network(expert_batch.observations)
    with torch.no_grad():
        next_q_values = target_q_network(expert_batch.next_observations)

    rewards = inverse_soft_q_reward(
        q_values=q_values,
        actions=expert_batch.actions,
        next_q_values=next_q_values,
        dones=expert_batch.dones,
        gamma=gamma,
        temperature=temperature,
    )
    replay_q_values = q_network(replay_batch.observations)
    with torch.no_grad():
        replay_next_q_values = target_q_network(replay_batch.next_observations)

    replay_values = soft_value_from_q(replay_q_values, temperature=temperature)
    replay_next_values = soft_value_from_q(replay_next_q_values, temperature=temperature)
    replay_term = (
        replay_values
        - gamma * (1.0 - replay_batch.dones.float()) * replay_next_values
    ).mean()

    expert_term = -rewards.mean()
    regularizer = chi2_regularizer(rewards)
    loss = expert_term + replay_term + regularization_weight * regularizer

    metrics = {
        "loss": loss.detach(),
        "expert_reward_mean": rewards.detach().mean(),
        "expert_term": expert_term.detach(),
        "replay_term": replay_term.detach(),
        "regularizer": regularizer.detach(),
    }
    return loss, metrics


def soft_policy_from_q(q_values: Tensor, temperature: float = 1.0) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.softmax(q_values / temperature, dim=-1)

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for h in hidden_dims:
        layers.extend((nn.Linear(current, h), nn.ReLU()))
        current = h
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class BCDiscretePolicy(nn.Module):
    """Behavioral cloning policy for discrete actions (softmax MLP).

    Trained via cross-entropy on expert demonstrations, then frozen.
    Provides the coherent reward signal: r(s,a) = alpha * log(pi_BC(a|s) / pi_uniform(a)).
    """

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.model = _build_mlp(observation_dim, num_actions, hidden_dims)
        self.num_actions = num_actions

    def forward(self, observations: Tensor) -> Tensor:
        return self.model(observations)  # logits [B, A]

    def log_probs(self, observations: Tensor) -> Tensor:
        return F.log_softmax(self.forward(observations), dim=-1)  # [B, A]


class BCContinuousActor(nn.Module):
    """Behavioral cloning policy for continuous actions (plain Gaussian, no tanh squash).

    Used for BC pretraining (frozen) and as the RL actor during fine-tuning.
    The coherent reward is: r(s,a) = alpha * (log pi_BC(a|s) - log N(a;0,I)).
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.backbone = _build_mlp(observation_dim, action_dim * 2, hidden_dims)
        self.action_dim = action_dim

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self.backbone(observations).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def log_prob_action(self, observations: Tensor, actions: Tensor) -> Tensor:
        """log p(action | obs) under the Gaussian policy."""
        mean, log_std = self(observations)
        return Normal(mean, log_std.exp()).log_prob(actions).sum(dim=-1)

    def sample(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self(observations)
        dist = Normal(mean, log_std.exp())
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def deterministic(self, observations: Tensor) -> Tensor:
        mean, _ = self(observations)
        return mean


# ──────────────────────────────────────────────────────────────────────────────
# BC training losses
# ──────────────────────────────────────────────────────────────────────────────


def bc_discrete_loss(
    observations: Tensor,
    actions: Tensor,
    bc_policy: BCDiscretePolicy,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Cross-entropy behavioral cloning loss for discrete actions."""
    logits = bc_policy(observations)
    loss = F.cross_entropy(logits, actions.long())
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.log().clamp(min=-20)).sum(-1).mean()
    return loss, {"bc_loss": loss.detach(), "bc_entropy": entropy}


def bc_continuous_loss(
    observations: Tensor,
    actions: Tensor,
    bc_actor: BCContinuousActor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Negative log-likelihood behavioral cloning loss for continuous actions."""
    log_prob = bc_actor.log_prob_action(observations, actions)
    loss = -log_prob.mean()
    return loss, {"bc_loss": loss.detach(), "bc_log_prob_mean": log_prob.detach().mean()}


# ──────────────────────────────────────────────────────────────────────────────
# Coherent reward (the core of CSIL: Watson et al., NeurIPS 2023)
# ──────────────────────────────────────────────────────────────────────────────


def coherent_reward_discrete(
    observations: Tensor,
    actions: Tensor,
    bc_policy: BCDiscretePolicy,
    alpha: float,
) -> Tensor:
    """Coherent reward for discrete actions.

    r(s,a) = alpha * log(pi_BC(a|s) / pi_uniform(a))
           = alpha * (log pi_BC(a|s) + log|A|)

    The uniform prior log(1/|A|) = -log|A| cancels to give +log|A|.
    """
    with torch.no_grad():
        log_probs = bc_policy.log_probs(observations)  # [B, A]
        log_pi_bc = log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        log_prior_correction = math.log(bc_policy.num_actions)
    return alpha * (log_pi_bc + log_prior_correction)


def coherent_reward_continuous(
    observations: Tensor,
    actions: Tensor,
    bc_actor: BCContinuousActor,
    alpha: float,
) -> Tensor:
    """Coherent reward for continuous actions.

    r(s,a) = alpha * (log pi_BC(a|s) - log N(a; 0, I))

    Uses a standard Normal prior N(0,I) over action dimensions.
    """
    with torch.no_grad():
        bc_log_prob = bc_actor.log_prob_action(observations, actions)
        prior_log_prob = Normal(
            torch.zeros_like(actions), torch.ones_like(actions)
        ).log_prob(actions).sum(dim=-1)
        log_ratio = bc_log_prob - prior_log_prob
    return alpha * log_ratio


# ──────────────────────────────────────────────────────────────────────────────
# Policy update losses (soft Q-learning / SAC-style)
# ──────────────────────────────────────────────────────────────────────────────


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
    """Soft Q-learning update using the CSIL coherent reward (discrete actions)."""
    with torch.no_grad():
        next_q = target_q_network(next_observations)
        next_values = temperature * torch.logsumexp(next_q / temperature, dim=-1)
        targets = rewards + gamma * (1.0 - dones) * next_values

    current_q = q_network(observations).gather(1, actions.long().unsqueeze(1)).squeeze(1)
    loss = F.mse_loss(current_q, targets)
    return loss, {
        "q_loss": loss.detach(),
        "q_mean": current_q.detach().mean(),
        "target_mean": targets.detach().mean(),
    }


def csil_continuous_critic_loss(
    observations: Tensor,
    actions: Tensor,
    next_observations: Tensor,
    dones: Tensor,
    rewards: Tensor,
    q_network: nn.Module,
    target_q_network: nn.Module,
    rl_actor: nn.Module,
    *,
    gamma: float = 0.99,
    temperature: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """SAC-style critic update using the CSIL coherent reward (continuous actions)."""
    with torch.no_grad():
        next_actions, next_log_probs = rl_actor.sample(next_observations)
        next_q = target_q_network(next_observations, next_actions)
        next_values = next_q - temperature * next_log_probs
        targets = rewards + gamma * (1.0 - dones) * next_values

    current_q = q_network(observations, actions)
    loss = F.mse_loss(current_q, targets)
    return loss, {
        "q_loss": loss.detach(),
        "q_mean": current_q.detach().mean(),
        "target_mean": targets.detach().mean(),
    }


def csil_actor_loss(
    observations: Tensor,
    q_network: nn.Module,
    rl_actor: nn.Module,
    *,
    temperature: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """SAC-style actor update (continuous actions)."""
    sampled_actions, log_probs = rl_actor.sample(observations)
    q_values = q_network(observations, sampled_actions)
    loss = (temperature * log_probs - q_values).mean()
    return loss, {"actor_loss": loss.detach(), "entropy": (-log_probs).detach().mean()}

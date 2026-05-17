from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.datasets.transitions import TransitionBatch
from src.imitation.base import ImitationAgent
from src.imitation.iqlearn import chi2_regularizer


def build_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.backbone = build_mlp(observation_dim, action_dim * 2, hidden_dims)
        self.action_dim = action_dim

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self.backbone(observations).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def sample(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self(observations)
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action) - torch.log(
            1 - action.pow(2) + 1e-6
        )
        return action, log_prob.sum(dim=-1)

    def deterministic(self, observations: Tensor) -> Tensor:
        mean, _ = self(observations)
        return torch.tanh(mean)


class ContinuousQNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.model = build_mlp(observation_dim + action_dim, 1, hidden_dims)

    def forward(self, observations: Tensor, actions: Tensor) -> Tensor:
        return self.model(torch.cat([observations, actions], dim=-1)).squeeze(-1)


class ContinuousIQLearnAgent(ImitationAgent):
    def __init__(
        self,
        actor: GaussianActor,
        q_network: ContinuousQNetwork,
        target_q_network: ContinuousQNetwork,
        *,
        gamma: float,
        temperature: float,
        regularization_weight: float,
        learning_rate: float,
        actor_learning_rate: float,
        target_update_interval: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.q_network = q_network.to(self.device)
        self.target_q_network = target_q_network.to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.gamma = gamma
        self.temperature = temperature
        self.regularization_weight = regularization_weight
        self.learning_rate = learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.target_update_interval = target_update_interval
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.update_steps = 0

    @classmethod
    def build_model(
        cls,
        observation_dim: int,
        action_dim: int,
        action_type: str,
        config: dict[str, Any],
    ) -> "ContinuousIQLearnAgent":
        if action_type != "continuous":
            raise ValueError("ContinuousIQLearnAgent only supports continuous action spaces")
        hidden_dims = tuple(config.get("hidden_dims", (256, 256)))
        return cls(
            actor=GaussianActor(observation_dim, action_dim, hidden_dims),
            q_network=ContinuousQNetwork(observation_dim, action_dim, hidden_dims),
            target_q_network=ContinuousQNetwork(observation_dim, action_dim, hidden_dims),
            gamma=float(config.get("gamma", 0.99)),
            temperature=float(config.get("temperature", 0.1)),
            regularization_weight=float(config.get("regularization_weight", 1.0)),
            learning_rate=float(config.get("learning_rate", 3e-4)),
            actor_learning_rate=float(config.get("actor_learning_rate", 3e-4)),
            target_update_interval=int(config.get("target_update_interval", 250)),
            device=config.get("device", "cpu"),
        )

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        raise NotImplementedError("Use update_with_replay for IQ-Learn training")

    def update_with_replay(
        self,
        expert_batch: TransitionBatch,
        replay_batch: TransitionBatch,
    ) -> dict[str, float]:
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(expert_batch.next_observations)
            next_values = self.target_q_network(
                expert_batch.next_observations, next_actions
            ) - self.temperature * next_log_probs
        rewards = self.q_network(expert_batch.observations, expert_batch.actions) - self.gamma * (
            1.0 - expert_batch.dones
        ) * next_values

        replay_actions, replay_log_probs = self.actor.sample(replay_batch.observations)
        replay_values = self.q_network(replay_batch.observations, replay_actions) - (
            self.temperature * replay_log_probs
        )
        with torch.no_grad():
            replay_next_actions, replay_next_log_probs = self.actor.sample(
                replay_batch.next_observations
            )
            replay_next_values = self.target_q_network(
                replay_batch.next_observations, replay_next_actions
            ) - self.temperature * replay_next_log_probs
        replay_term = (
            replay_values - self.gamma * (1.0 - replay_batch.dones) * replay_next_values
        ).mean()
        expert_term = -rewards.mean()
        regularizer = chi2_regularizer(rewards)
        critic_loss = expert_term + replay_term + self.regularization_weight * regularizer

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        sampled_actions, log_probs = self.actor.sample(replay_batch.observations)
        actor_loss = (
            self.temperature * log_probs
            - self.q_network(replay_batch.observations, sampled_actions)
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        with torch.no_grad():
            q_values = self.q_network(expert_batch.observations, expert_batch.actions)

        return {
            "loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "expert_reward_mean": float(rewards.mean().item()),
            "expert_term": float(expert_term.item()),
            "replay_term": float(replay_term.item()),
            "regularizer": float(regularizer.item()),
            "q_mean": float(q_values.mean().item()),
            "q_abs_max": float(q_values.abs().max().item()),
        }

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(observation_tensor)
            else:
                action, _ = self.actor.sample(observation_tensor)
        return action.squeeze(0).cpu().numpy()

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "q_network_state_dict": self.q_network.state_dict(),
                "target_q_network_state_dict": self.target_q_network.state_dict(),
                "gamma": self.gamma,
                "temperature": self.temperature,
                "regularization_weight": self.regularization_weight,
                "learning_rate": self.learning_rate,
                "actor_learning_rate": self.actor_learning_rate,
                "target_update_interval": self.target_update_interval,
                "observation_dim": self.actor.backbone[0].in_features,
                "action_dim": self.actor.action_dim,
                "hidden_dims": [
                    layer.out_features
                    for layer in self.actor.backbone
                    if isinstance(layer, torch.nn.Linear)
                ][:-1],
            },
            output_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ContinuousIQLearnAgent":
        payload = torch.load(path, map_location="cpu")
        config = {
            "gamma": payload["gamma"],
            "temperature": payload["temperature"],
            "regularization_weight": payload["regularization_weight"],
            "learning_rate": payload["learning_rate"],
            "actor_learning_rate": payload["actor_learning_rate"],
            "target_update_interval": payload["target_update_interval"],
            "hidden_dims": payload["hidden_dims"],
        }
        agent = cls.build_model(
            payload["observation_dim"],
            payload["action_dim"],
            "continuous",
            config,
        )
        agent.actor.load_state_dict(payload["actor_state_dict"])
        agent.q_network.load_state_dict(payload["q_network_state_dict"])
        agent.target_q_network.load_state_dict(payload["target_q_network_state_dict"])
        return agent


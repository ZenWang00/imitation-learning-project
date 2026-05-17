from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.datasets.transitions import TransitionBatch
from src.imitation.base import ImitationAgent
from src.imitation.iqlearn import (
    DiscreteQNetwork,
    IQBatch,
    iq_critic_loss,
    soft_policy_from_q,
)


class IQLearnAgent(ImitationAgent):
    def __init__(
        self,
        q_network: DiscreteQNetwork,
        target_q_network: DiscreteQNetwork,
        *,
        gamma: float,
        temperature: float,
        regularization_weight: float,
        learning_rate: float,
        target_update_interval: int,
        max_grad_norm: float | None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.q_network = q_network.to(self.device)
        self.target_q_network = target_q_network.to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.gamma = gamma
        self.temperature = temperature
        self.regularization_weight = regularization_weight
        self.learning_rate = learning_rate
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.update_steps = 0

    @classmethod
    def build_model(
        cls,
        observation_dim: int,
        action_dim: int,
        action_type: str,
        config: dict[str, Any],
    ) -> "IQLearnAgent":
        if action_type != "discrete":
            raise ValueError("IQLearnAgent only supports discrete action spaces")
        hidden_dims = tuple(config.get("hidden_dims", (256, 256)))
        q_network = DiscreteQNetwork(observation_dim, action_dim, hidden_dims=hidden_dims)
        target_q_network = DiscreteQNetwork(observation_dim, action_dim, hidden_dims=hidden_dims)
        return cls(
            q_network=q_network,
            target_q_network=target_q_network,
            gamma=float(config.get("gamma", 0.99)),
            temperature=float(config.get("temperature", 1.0)),
            regularization_weight=float(config.get("regularization_weight", 1.0)),
            learning_rate=float(config.get("learning_rate", 3e-4)),
            target_update_interval=int(config.get("target_update_interval", 100)),
            max_grad_norm=(
                float(config["max_grad_norm"])
                if config.get("max_grad_norm") is not None
                else None
            ),
            device=config.get("device", "cpu"),
        )

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        raise NotImplementedError("Use update_with_replay for IQ-Learn training")

    def update_with_replay(
        self,
        expert_batch: TransitionBatch,
        replay_batch: TransitionBatch,
    ) -> dict[str, float]:
        expert_iq_batch = IQBatch(
            observations=expert_batch.observations,
            actions=expert_batch.actions,
            next_observations=expert_batch.next_observations,
            dones=expert_batch.dones,
        )
        replay_iq_batch = IQBatch(
            observations=replay_batch.observations,
            actions=replay_batch.actions,
            next_observations=replay_batch.next_observations,
            dones=replay_batch.dones,
        )
        loss, metrics = iq_critic_loss(
            expert_iq_batch,
            replay_iq_batch,
            self.q_network,
            self.target_q_network,
            gamma=self.gamma,
            temperature=self.temperature,
            regularization_weight=self.regularization_weight,
        )
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        with torch.no_grad():
            q_values = self.q_network(expert_batch.observations)
            metrics["q_mean"] = q_values.mean().detach()
            metrics["q_abs_max"] = q_values.abs().max().detach()

        return {name: float(value.item()) for name, value in metrics.items()}

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(observation_tensor)
            if deterministic:
                return int(q_values.argmax(dim=-1).item())
            probabilities = soft_policy_from_q(q_values, temperature=self.temperature)
            return int(torch.multinomial(probabilities.squeeze(0), num_samples=1).item())

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_q_network_state_dict": self.target_q_network.state_dict(),
                "gamma": self.gamma,
                "temperature": self.temperature,
                "regularization_weight": self.regularization_weight,
                "learning_rate": self.learning_rate,
                "target_update_interval": self.target_update_interval,
                "max_grad_norm": self.max_grad_norm,
                "observation_dim": self.q_network.model[0].in_features,
                "action_dim": self.q_network.model[-1].out_features,
                "hidden_dims": [
                    layer.out_features
                    for layer in self.q_network.model
                    if isinstance(layer, torch.nn.Linear)
                ][:-1],
            },
            output_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "IQLearnAgent":
        payload = torch.load(path, map_location="cpu")
        config = {
            "gamma": payload["gamma"],
            "temperature": payload["temperature"],
            "regularization_weight": payload["regularization_weight"],
            "learning_rate": payload["learning_rate"],
            "target_update_interval": payload["target_update_interval"],
            "max_grad_norm": payload.get("max_grad_norm"),
            "hidden_dims": payload["hidden_dims"],
        }
        agent = cls.build_model(
            payload["observation_dim"],
            payload["action_dim"],
            "discrete",
            config,
        )
        agent.q_network.load_state_dict(payload["q_network_state_dict"])
        agent.target_q_network.load_state_dict(payload["target_q_network_state_dict"])
        return agent

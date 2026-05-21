from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.datasets.transitions import TransitionBatch, TransitionDataset
from src.imitation.base import ImitationAgent
from src.imitation.csil import (
    BCDiscretePolicy,
    bc_discrete_loss,
    coherent_reward_discrete,
    csil_discrete_q_loss,
)
from src.imitation.iqlearn import DiscreteQNetwork, soft_policy_from_q


class CSILAgent(ImitationAgent):
    """Coherent Soft Imitation Learning with discrete actions (CartPole-v1).

    Implements Watson et al., NeurIPS 2023.

    Training has two phases:
      1. BC pretraining: fit pi_BC to expert demonstrations via cross-entropy.
      2. RL fine-tuning: soft Q-learning with the coherent reward
         r(s,a) = alpha * (log pi_BC(a|s) + log|A|).

    The BC policy is frozen after pretraining and only used for reward computation.
    """

    def __init__(
        self,
        bc_policy: BCDiscretePolicy,
        q_network: DiscreteQNetwork,
        target_q_network: DiscreteQNetwork,
        *,
        gamma: float,
        temperature: float,
        bc_lr: float,
        policy_lr: float,
        target_update_interval: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.bc_policy = bc_policy.to(self.device)
        self.q_network = q_network.to(self.device)
        self.target_q_network = target_q_network.to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.gamma = gamma
        self.temperature = temperature
        self.bc_lr = bc_lr
        self.policy_lr = policy_lr
        self.target_update_interval = target_update_interval
        self.bc_optimizer = torch.optim.Adam(self.bc_policy.parameters(), lr=bc_lr)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=policy_lr)
        self.update_steps = 0
        self._bc_pretrained = False

    @classmethod
    def build_model(
        cls,
        observation_dim: int,
        action_dim: int,
        action_type: str,
        config: dict[str, Any],
    ) -> "CSILAgent":
        if action_type != "discrete":
            raise ValueError("CSILAgent only supports discrete action spaces")
        hidden_dims = tuple(config.get("hidden_dims", (128, 128)))
        return cls(
            bc_policy=BCDiscretePolicy(observation_dim, action_dim, hidden_dims),
            q_network=DiscreteQNetwork(observation_dim, action_dim, hidden_dims=hidden_dims),
            target_q_network=DiscreteQNetwork(
                observation_dim, action_dim, hidden_dims=hidden_dims
            ),
            gamma=float(config.get("gamma", 0.99)),
            temperature=float(config.get("temperature", 1.0)),
            bc_lr=float(config.get("bc_lr", 3e-4)),
            policy_lr=float(config.get("policy_lr", 3e-4)),
            target_update_interval=int(config.get("target_update_interval", 100)),
            device=config.get("device", "cpu"),
        )

    def pretrain(
        self,
        expert_dataset: TransitionDataset,
        num_steps: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> None:
        """Phase 1: train BC policy on expert demonstrations, then freeze it."""
        self.bc_policy.train()
        for step in range(1, num_steps + 1):
            batch = expert_dataset.sample_batch(batch_size, generator=rng, device=self.device)
            loss, _ = bc_discrete_loss(batch.observations, batch.actions, self.bc_policy)
            self.bc_optimizer.zero_grad()
            loss.backward()
            self.bc_optimizer.step()
            if step % max(1, num_steps // 5) == 0:
                print(f"  BC pretrain step {step}/{num_steps}  loss={loss.item():.4f}", flush=True)

        self.bc_policy.eval()
        self.bc_policy.requires_grad_(False)
        self._bc_pretrained = True

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        raise NotImplementedError("Use update_with_replay for CSIL training")

    def update_with_replay(
        self,
        expert_batch: TransitionBatch,
        replay_batch: TransitionBatch,
    ) -> dict[str, float]:
        rewards = coherent_reward_discrete(
            replay_batch.observations,
            replay_batch.actions,
            self.bc_policy,
            alpha=self.temperature,
        )

        q_loss, q_metrics = csil_discrete_q_loss(
            replay_batch.observations,
            replay_batch.actions,
            replay_batch.next_observations,
            replay_batch.dones,
            rewards,
            self.q_network,
            self.target_q_network,
            gamma=self.gamma,
            temperature=self.temperature,
        )
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        expert_rewards = coherent_reward_discrete(
            expert_batch.observations,
            expert_batch.actions,
            self.bc_policy,
            alpha=self.temperature,
        )
        return {
            "loss": float(q_loss.item()),
            "expert_reward_mean": float(expert_rewards.mean().item()),
            "expert_term": 0.0,
            "replay_term": 0.0,
            "regularizer": 0.0,
            "q_mean": float(q_metrics["q_mean"].item()),
            "q_abs_max": float(
                self.q_network(replay_batch.observations).abs().max().item()
            ),
        }

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
            if deterministic:
                return int(q_values.argmax(dim=-1).item())
            probs = soft_policy_from_q(q_values, temperature=self.temperature)
            return int(torch.multinomial(probs.squeeze(0), num_samples=1).item())

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bc_policy_state_dict": self.bc_policy.state_dict(),
                "q_network_state_dict": self.q_network.state_dict(),
                "target_q_network_state_dict": self.target_q_network.state_dict(),
                "gamma": self.gamma,
                "temperature": self.temperature,
                "bc_lr": self.bc_lr,
                "policy_lr": self.policy_lr,
                "target_update_interval": self.target_update_interval,
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
    def load(cls, path: str | Path) -> "CSILAgent":
        payload = torch.load(path, map_location="cpu")
        config = {
            "gamma": payload["gamma"],
            "temperature": payload["temperature"],
            "bc_lr": payload["bc_lr"],
            "policy_lr": payload["policy_lr"],
            "target_update_interval": payload["target_update_interval"],
            "hidden_dims": payload["hidden_dims"],
        }
        agent = cls.build_model(
            payload["observation_dim"], payload["action_dim"], "discrete", config
        )
        agent.bc_policy.load_state_dict(payload["bc_policy_state_dict"])
        agent.bc_policy.eval()
        agent.bc_policy.requires_grad_(False)
        agent.q_network.load_state_dict(payload["q_network_state_dict"])
        agent.target_q_network.load_state_dict(payload["target_q_network_state_dict"])
        return agent

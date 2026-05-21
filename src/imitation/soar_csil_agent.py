from __future__ import annotations

import math
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


class SOARCSILAgent(ImitationAgent):
    """SOAR-enhanced CSIL for discrete actions (CartPole-v1).

    Implements the SOAR template from Viel et al., ICML 2025 (arXiv:2502.19859)
    applied on top of Coherent Soft Imitation Learning.

    SOAR = Soft Optimistic Actor cRitic: uses an ensemble of N Q-networks.
    The optimistic Q-value used for the policy is:
        Q_opt(s,a) = mean_i[Q_i(s,a)] + uncertainty_coef * std_i[Q_i(s,a)]

    This gives an exploration bonus proportional to Q-value disagreement across
    the ensemble, encouraging the policy to visit under-explored regions.
    """

    def __init__(
        self,
        bc_policy: BCDiscretePolicy,
        q_networks: list[DiscreteQNetwork],
        target_q_networks: list[DiscreteQNetwork],
        *,
        gamma: float,
        temperature: float,
        uncertainty_coef: float,
        q_std_clip: float,
        bc_lr: float,
        policy_lr: float,
        target_update_interval: int,
        device: torch.device | str = "cpu",
    ) -> None:
        assert len(q_networks) == len(target_q_networks) and len(q_networks) >= 1
        self.device = torch.device(device)
        self.bc_policy = bc_policy.to(self.device)
        self.q_networks = [q.to(self.device) for q in q_networks]
        self.target_q_networks = [q.to(self.device) for q in target_q_networks]
        for q, tq in zip(self.q_networks, self.target_q_networks):
            tq.load_state_dict(q.state_dict())
        self.gamma = gamma
        self.temperature = temperature
        self.uncertainty_coef = uncertainty_coef
        self.q_std_clip = q_std_clip
        self.bc_lr = bc_lr
        self.policy_lr = policy_lr
        self.target_update_interval = target_update_interval
        self.bc_optimizer = torch.optim.Adam(self.bc_policy.parameters(), lr=bc_lr)
        self.q_optimizers = [
            torch.optim.Adam(q.parameters(), lr=policy_lr) for q in self.q_networks
        ]
        self.update_steps = 0

    @classmethod
    def build_model(
        cls,
        observation_dim: int,
        action_dim: int,
        action_type: str,
        config: dict[str, Any],
    ) -> "SOARCSILAgent":
        if action_type != "discrete":
            raise ValueError("SOARCSILAgent only supports discrete action spaces")
        hidden_dims = tuple(config.get("hidden_dims", (128, 128)))
        num_q = int(config.get("num_q_networks", 4))
        q_networks = [
            DiscreteQNetwork(observation_dim, action_dim, hidden_dims=hidden_dims)
            for _ in range(num_q)
        ]
        target_q_networks = [
            DiscreteQNetwork(observation_dim, action_dim, hidden_dims=hidden_dims)
            for _ in range(num_q)
        ]
        return cls(
            bc_policy=BCDiscretePolicy(observation_dim, action_dim, hidden_dims),
            q_networks=q_networks,
            target_q_networks=target_q_networks,
            gamma=float(config.get("gamma", 0.99)),
            temperature=float(config.get("temperature", 1.0)),
            uncertainty_coef=float(config.get("uncertainty_coef", 1.0)),
            q_std_clip=float(config.get("q_std_clip", 1.0)),
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

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        raise NotImplementedError("Use update_with_replay for SOAR-CSIL training")

    def _optimistic_q(self, observations: torch.Tensor) -> torch.Tensor:
        """Q_opt(s,a) = mean_i[Q_i(s,a)] + uncertainty_coef * std_i[Q_i(s,a)]."""
        q_stack = torch.stack(
            [q(observations) for q in self.q_networks], dim=0
        )  # [N, B, A]
        q_mean = q_stack.mean(dim=0)
        if len(self.q_networks) > 1:
            q_std = q_stack.std(dim=0).clamp(0.0, self.q_std_clip)
            return q_mean + self.uncertainty_coef * q_std
        return q_mean

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

        # Update each Q-network independently (different random mini-batches would be
        # ideal, but here both use the same replay_batch for simplicity)
        total_q_loss = 0.0
        for q_net, target_q_net, opt in zip(
            self.q_networks, self.target_q_networks, self.q_optimizers
        ):
            q_loss, q_metrics = csil_discrete_q_loss(
                replay_batch.observations,
                replay_batch.actions,
                replay_batch.next_observations,
                replay_batch.dones,
                rewards,
                q_net,
                target_q_net,
                gamma=self.gamma,
                temperature=self.temperature,
            )
            opt.zero_grad()
            q_loss.backward()
            opt.step()
            total_q_loss += q_loss.item()

        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
                target_q_net.load_state_dict(q_net.state_dict())

        expert_rewards = coherent_reward_discrete(
            expert_batch.observations,
            expert_batch.actions,
            self.bc_policy,
            alpha=self.temperature,
        )

        with torch.no_grad():
            q_opt = self._optimistic_q(replay_batch.observations)

        return {
            "loss": total_q_loss / len(self.q_networks),
            "expert_reward_mean": float(expert_rewards.mean().item()),
            "expert_term": 0.0,
            "replay_term": 0.0,
            "regularizer": 0.0,
            "q_mean": float(q_opt.mean().item()),
            "q_abs_max": float(q_opt.abs().max().item()),
        }

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_opt = self._optimistic_q(obs_tensor)
            if deterministic:
                return int(q_opt.argmax(dim=-1).item())
            probs = soft_policy_from_q(q_opt, temperature=self.temperature)
            return int(torch.multinomial(probs.squeeze(0), num_samples=1).item())

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hidden_dims = [
            layer.out_features
            for layer in self.q_networks[0].model
            if isinstance(layer, torch.nn.Linear)
        ][:-1]
        torch.save(
            {
                "bc_policy_state_dict": self.bc_policy.state_dict(),
                "q_network_state_dicts": [q.state_dict() for q in self.q_networks],
                "target_q_network_state_dicts": [
                    q.state_dict() for q in self.target_q_networks
                ],
                "gamma": self.gamma,
                "temperature": self.temperature,
                "uncertainty_coef": self.uncertainty_coef,
                "q_std_clip": self.q_std_clip,
                "bc_lr": self.bc_lr,
                "policy_lr": self.policy_lr,
                "target_update_interval": self.target_update_interval,
                "observation_dim": self.q_networks[0].model[0].in_features,
                "action_dim": self.q_networks[0].model[-1].out_features,
                "hidden_dims": hidden_dims,
                "num_q_networks": len(self.q_networks),
            },
            output_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SOARCSILAgent":
        payload = torch.load(path, map_location="cpu")
        config = {
            "gamma": payload["gamma"],
            "temperature": payload["temperature"],
            "uncertainty_coef": payload["uncertainty_coef"],
            "q_std_clip": payload["q_std_clip"],
            "bc_lr": payload["bc_lr"],
            "policy_lr": payload["policy_lr"],
            "target_update_interval": payload["target_update_interval"],
            "hidden_dims": payload["hidden_dims"],
            "num_q_networks": payload["num_q_networks"],
        }
        agent = cls.build_model(
            payload["observation_dim"], payload["action_dim"], "discrete", config
        )
        agent.bc_policy.load_state_dict(payload["bc_policy_state_dict"])
        agent.bc_policy.eval()
        agent.bc_policy.requires_grad_(False)
        for q, sd in zip(agent.q_networks, payload["q_network_state_dicts"]):
            q.load_state_dict(sd)
        for q, sd in zip(agent.target_q_networks, payload["target_q_network_state_dicts"]):
            q.load_state_dict(sd)
        return agent

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.datasets.transitions import TransitionBatch, TransitionDataset
from src.imitation.base import ImitationAgent
from src.imitation.continuous_iqlearn_agent import ContinuousQNetwork
from src.imitation.csil import (
    BCContinuousActor,
    bc_continuous_loss,
    coherent_reward_continuous,
    csil_continuous_critic_loss,
)


class ContinuousSOARCSILAgent(ImitationAgent):
    """SOAR-enhanced CSIL for continuous actions (Pendulum-v1).

    Implements the SOAR template from Viel et al., ICML 2025 (arXiv:2502.19859)
    applied on top of Coherent Soft Imitation Learning.

    SOAR = Soft Optimistic Actor cRitic: N Q-networks. The actor maximises
        Q_opt(s,a) = mean_i[Q_i(s,a)] + uncertainty_coef * std_i[Q_i(s,a)]
    yielding a UCB-style exploration bonus from Q-value disagreement.

    Two-phase training:
      1. BC pretraining: fit bc_actor via NLL on expert data, then freeze it.
      2. RL fine-tuning: SAC with coherent reward r = alpha*(log pi_BC - log N(0,I)).
         rl_actor warm-started from bc_actor, then trained with the optimistic Q actor loss.
    """

    def __init__(
        self,
        bc_actor: BCContinuousActor,
        rl_actor: BCContinuousActor,
        q_networks: list[ContinuousQNetwork],
        target_q_networks: list[ContinuousQNetwork],
        *,
        gamma: float,
        temperature: float,
        uncertainty_coef: float,
        q_std_clip: float,
        bc_lr: float,
        policy_lr: float,
        actor_lr: float,
        target_update_interval: int,
        device: torch.device | str = "cpu",
    ) -> None:
        assert len(q_networks) == len(target_q_networks) and len(q_networks) >= 1
        self.device = torch.device(device)
        self.bc_actor = bc_actor.to(self.device)
        self.rl_actor = rl_actor.to(self.device)
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
        self.actor_lr = actor_lr
        self.target_update_interval = target_update_interval
        self.bc_optimizer = torch.optim.Adam(self.bc_actor.parameters(), lr=bc_lr)
        self.q_optimizers = [
            torch.optim.Adam(q.parameters(), lr=policy_lr) for q in self.q_networks
        ]
        self.actor_optimizer = torch.optim.Adam(self.rl_actor.parameters(), lr=actor_lr)
        self.update_steps = 0

    @classmethod
    def build_model(
        cls,
        observation_dim: int,
        action_dim: int,
        action_type: str,
        config: dict[str, Any],
    ) -> "ContinuousSOARCSILAgent":
        if action_type != "continuous":
            raise ValueError("ContinuousSOARCSILAgent only supports continuous action spaces")
        hidden_dims = tuple(config.get("hidden_dims", (256, 256)))
        num_q = int(config.get("num_q_networks", 4))
        q_networks = [
            ContinuousQNetwork(observation_dim, action_dim, hidden_dims=hidden_dims)
            for _ in range(num_q)
        ]
        target_q_networks = [
            ContinuousQNetwork(observation_dim, action_dim, hidden_dims=hidden_dims)
            for _ in range(num_q)
        ]
        return cls(
            bc_actor=BCContinuousActor(observation_dim, action_dim, hidden_dims),
            rl_actor=BCContinuousActor(observation_dim, action_dim, hidden_dims),
            q_networks=q_networks,
            target_q_networks=target_q_networks,
            gamma=float(config.get("gamma", 0.99)),
            temperature=float(config.get("temperature", 0.1)),
            uncertainty_coef=float(config.get("uncertainty_coef", 1.0)),
            q_std_clip=float(config.get("q_std_clip", 1.0)),
            bc_lr=float(config.get("bc_lr", 3e-4)),
            policy_lr=float(config.get("policy_lr", 3e-4)),
            actor_lr=float(config.get("actor_lr", 3e-4)),
            target_update_interval=int(config.get("target_update_interval", 250)),
            device=config.get("device", "cpu"),
        )

    def pretrain(
        self,
        expert_dataset: TransitionDataset,
        num_steps: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> None:
        """Phase 1: train bc_actor on expert demonstrations, then freeze it."""
        self.bc_actor.train()
        for step in range(1, num_steps + 1):
            batch = expert_dataset.sample_batch(batch_size, generator=rng, device=self.device)
            loss, _ = bc_continuous_loss(batch.observations, batch.actions, self.bc_actor)
            self.bc_optimizer.zero_grad()
            loss.backward()
            self.bc_optimizer.step()
            if step % max(1, num_steps // 5) == 0:
                print(f"  BC pretrain step {step}/{num_steps}  loss={loss.item():.4f}", flush=True)

        self.bc_actor.eval()
        self.bc_actor.requires_grad_(False)

        self.rl_actor.load_state_dict(self.bc_actor.state_dict())
        self.rl_actor.requires_grad_(True)
        self.rl_actor.train()
        self.actor_optimizer = torch.optim.Adam(self.rl_actor.parameters(), lr=self.actor_lr)

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        raise NotImplementedError("Use update_with_replay for SOAR-CSIL training")

    def _optimistic_q(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Q_opt(s,a) = mean_i[Q_i(s,a)] + uncertainty_coef * std_i[Q_i(s,a)]."""
        q_vals = torch.stack(
            [q(observations, actions) for q in self.q_networks], dim=0
        )  # [N, B]
        q_mean = q_vals.mean(dim=0)
        if len(self.q_networks) > 1:
            q_std = q_vals.std(dim=0).clamp(0.0, self.q_std_clip)
            return q_mean + self.uncertainty_coef * q_std
        return q_mean

    def update_with_replay(
        self,
        expert_batch: TransitionBatch,
        replay_batch: TransitionBatch,
    ) -> dict[str, float]:
        rewards = coherent_reward_continuous(
            replay_batch.observations,
            replay_batch.actions,
            self.bc_actor,
            alpha=self.temperature,
        )

        # Update each Q-network independently with its own target
        total_q_loss = 0.0
        for q_net, target_q_net, opt in zip(
            self.q_networks, self.target_q_networks, self.q_optimizers
        ):
            q_loss, q_metrics = csil_continuous_critic_loss(
                replay_batch.observations,
                replay_batch.actions,
                replay_batch.next_observations,
                replay_batch.dones,
                rewards,
                q_net,
                target_q_net,
                self.rl_actor,
                gamma=self.gamma,
                temperature=self.temperature,
            )
            opt.zero_grad()
            q_loss.backward()
            opt.step()
            total_q_loss += q_loss.item()

        # Actor update: maximise optimistic Q minus entropy penalty
        sampled_actions, log_probs = self.rl_actor.sample(replay_batch.observations)
        q_opt = self._optimistic_q(replay_batch.observations, sampled_actions)
        actor_loss = (self.temperature * log_probs - q_opt).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
                target_q_net.load_state_dict(q_net.state_dict())

        expert_rewards = coherent_reward_continuous(
            expert_batch.observations,
            expert_batch.actions,
            self.bc_actor,
            alpha=self.temperature,
        )
        with torch.no_grad():
            q_opt_stats = self._optimistic_q(
                replay_batch.observations, replay_batch.actions
            )
        return {
            "loss": total_q_loss / len(self.q_networks),
            "actor_loss": float(actor_loss.item()),
            "expert_reward_mean": float(expert_rewards.mean().item()),
            "expert_term": 0.0,
            "replay_term": 0.0,
            "regularizer": 0.0,
            "q_mean": float(q_opt_stats.mean().item()),
            "q_abs_max": float(q_opt_stats.abs().max().item()),
        }

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.rl_actor.deterministic(obs_tensor)
            else:
                action, _ = self.rl_actor.sample(obs_tensor)
        return action.squeeze(0).cpu().numpy()

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hidden_dims = [
            layer.out_features
            for layer in self.bc_actor.backbone
            if isinstance(layer, torch.nn.Linear)
        ][:-1]
        torch.save(
            {
                "bc_actor_state_dict": self.bc_actor.state_dict(),
                "rl_actor_state_dict": self.rl_actor.state_dict(),
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
                "actor_lr": self.actor_lr,
                "target_update_interval": self.target_update_interval,
                "observation_dim": self.bc_actor.backbone[0].in_features,
                "action_dim": self.bc_actor.action_dim,
                "hidden_dims": hidden_dims,
                "num_q_networks": len(self.q_networks),
            },
            output_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ContinuousSOARCSILAgent":
        payload = torch.load(path, map_location="cpu")
        config = {
            "gamma": payload["gamma"],
            "temperature": payload["temperature"],
            "uncertainty_coef": payload["uncertainty_coef"],
            "q_std_clip": payload["q_std_clip"],
            "bc_lr": payload["bc_lr"],
            "policy_lr": payload["policy_lr"],
            "actor_lr": payload["actor_lr"],
            "target_update_interval": payload["target_update_interval"],
            "hidden_dims": payload["hidden_dims"],
            "num_q_networks": payload["num_q_networks"],
        }
        agent = cls.build_model(
            payload["observation_dim"], payload["action_dim"], "continuous", config
        )
        agent.bc_actor.load_state_dict(payload["bc_actor_state_dict"])
        agent.bc_actor.eval()
        agent.bc_actor.requires_grad_(False)
        agent.rl_actor.load_state_dict(payload["rl_actor_state_dict"])
        for q, sd in zip(agent.q_networks, payload["q_network_state_dicts"]):
            q.load_state_dict(sd)
        for q, sd in zip(agent.target_q_networks, payload["target_q_network_state_dicts"]):
            q.load_state_dict(sd)
        return agent

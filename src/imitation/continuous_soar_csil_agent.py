from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.datasets.transitions import TransitionBatch
from src.imitation.base import ImitationAgent
from src.imitation.continuous_iqlearn_agent import ContinuousQNetwork, GaussianActor
from src.imitation.csil import (
    RewardNetwork,
    csil_actor_loss,
    csil_continuous_critic_loss,
    csil_reward_loss,
)
from src.imitation.soar import StateDiscriminator, soar_discriminator_loss


class ContinuousSOARCSILAgent(ImitationAgent):
    """SOAR-enhanced CSIL with continuous actions (Pendulum-v1).

    Extends ContinuousCSILAgent with a state occupancy discriminator (SOAR).
    The total reward is: r_csil(s,a) + soar_weight * log_odds_D(s).
    """

    def __init__(
        self,
        reward_network: RewardNetwork,
        discriminator: StateDiscriminator,
        actor: GaussianActor,
        q_network: ContinuousQNetwork,
        target_q_network: ContinuousQNetwork,
        *,
        gamma: float,
        temperature: float,
        regularization_weight: float,
        soar_weight: float,
        reward_lr: float,
        discriminator_lr: float,
        policy_lr: float,
        actor_lr: float,
        target_update_interval: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.reward_network = reward_network.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.actor = actor.to(self.device)
        self.q_network = q_network.to(self.device)
        self.target_q_network = target_q_network.to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.gamma = gamma
        self.temperature = temperature
        self.regularization_weight = regularization_weight
        self.soar_weight = soar_weight
        self.reward_lr = reward_lr
        self.discriminator_lr = discriminator_lr
        self.policy_lr = policy_lr
        self.actor_lr = actor_lr
        self.target_update_interval = target_update_interval
        self.reward_optimizer = torch.optim.Adam(
            self.reward_network.parameters(), lr=reward_lr
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=discriminator_lr
        )
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=policy_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
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
            raise ValueError(
                "ContinuousSOARCSILAgent only supports continuous action spaces"
            )
        hidden_dims = tuple(config.get("hidden_dims", (256, 256)))
        soar_hidden_dims = tuple(config.get("soar_hidden_dims", (128, 128)))
        return cls(
            reward_network=RewardNetwork(
                input_dim=observation_dim + action_dim,
                hidden_dims=hidden_dims,
            ),
            discriminator=StateDiscriminator(
                observation_dim=observation_dim,
                hidden_dims=soar_hidden_dims,
            ),
            actor=GaussianActor(observation_dim, action_dim, hidden_dims),
            q_network=ContinuousQNetwork(observation_dim, action_dim, hidden_dims),
            target_q_network=ContinuousQNetwork(observation_dim, action_dim, hidden_dims),
            gamma=float(config.get("gamma", 0.99)),
            temperature=float(config.get("temperature", 0.1)),
            regularization_weight=float(config.get("regularization_weight", 0.25)),
            soar_weight=float(config.get("soar_weight", 0.1)),
            reward_lr=float(config.get("reward_lr", 3e-4)),
            discriminator_lr=float(config.get("discriminator_lr", 3e-4)),
            policy_lr=float(config.get("policy_lr", 3e-4)),
            actor_lr=float(config.get("actor_lr", 3e-4)),
            target_update_interval=int(config.get("target_update_interval", 250)),
            device=config.get("device", "cpu"),
        )

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        raise NotImplementedError("Use update_with_replay for SOAR-CSIL training")

    def update_with_replay(
        self,
        expert_batch: TransitionBatch,
        replay_batch: TransitionBatch,
    ) -> dict[str, float]:
        # 1. Update CSIL reward network
        expert_features = torch.cat(
            [expert_batch.observations, expert_batch.actions], dim=-1
        )
        replay_features = torch.cat(
            [replay_batch.observations, replay_batch.actions], dim=-1
        )
        reward_loss, reward_metrics = csil_reward_loss(
            expert_features,
            replay_features,
            self.reward_network,
            regularization_weight=self.regularization_weight,
        )
        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()

        # 2. Update SOAR state discriminator
        disc_loss, disc_metrics = soar_discriminator_loss(
            expert_batch.observations,
            replay_batch.observations,
            self.discriminator,
        )
        self.discriminator_optimizer.zero_grad()
        disc_loss.backward()
        self.discriminator_optimizer.step()

        # 3. Compute combined reward
        with torch.no_grad():
            csil_rewards = self.reward_network(
                torch.cat([replay_batch.observations, replay_batch.actions], dim=-1)
            )
            soar_bonus = self.discriminator.log_odds(replay_batch.observations)
            total_rewards = csil_rewards + self.soar_weight * soar_bonus

        # 4. Update critic
        q_loss, q_metrics = csil_continuous_critic_loss(
            replay_batch.observations,
            replay_batch.actions,
            replay_batch.next_observations,
            replay_batch.dones,
            total_rewards,
            self.q_network,
            self.target_q_network,
            self.actor,
            gamma=self.gamma,
            temperature=self.temperature,
        )
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # 5. Update actor
        actor_loss, _ = csil_actor_loss(
            replay_batch.observations,
            self.q_network,
            self.actor,
            temperature=self.temperature,
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "expert_reward_mean": float(reward_metrics["expert_reward_mean"].item()),
            "expert_term": float((-reward_metrics["expert_reward_mean"]).item()),
            "replay_term": float(reward_metrics["replay_reward_mean"].item()),
            "regularizer": float(reward_metrics["regularizer"].item()),
            "q_mean": float(q_metrics["q_mean"].item()),
            "q_abs_max": float(q_metrics["q_mean"].abs().item()),
            "soar_bonus_mean": float(soar_bonus.mean().item()),
            "soar_disc_loss": float(disc_metrics["soar_disc_loss"].item()),
        }

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(obs_tensor)
            else:
                action, _ = self.actor.sample(obs_tensor)
        return action.squeeze(0).cpu().numpy()

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "reward_network_state_dict": self.reward_network.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "actor_state_dict": self.actor.state_dict(),
                "q_network_state_dict": self.q_network.state_dict(),
                "target_q_network_state_dict": self.target_q_network.state_dict(),
                "gamma": self.gamma,
                "temperature": self.temperature,
                "regularization_weight": self.regularization_weight,
                "soar_weight": self.soar_weight,
                "reward_lr": self.reward_lr,
                "discriminator_lr": self.discriminator_lr,
                "policy_lr": self.policy_lr,
                "actor_lr": self.actor_lr,
                "target_update_interval": self.target_update_interval,
                "observation_dim": self.actor.backbone[0].in_features,
                "action_dim": self.actor.action_dim,
                "hidden_dims": [
                    layer.out_features
                    for layer in self.actor.backbone
                    if isinstance(layer, torch.nn.Linear)
                ][:-1],
                "soar_hidden_dims": [
                    layer.out_features
                    for layer in self.discriminator.model
                    if isinstance(layer, torch.nn.Linear)
                ][:-1],
            },
            output_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ContinuousSOARCSILAgent":
        payload = torch.load(path, map_location="cpu")
        config = {
            "gamma": payload["gamma"],
            "temperature": payload["temperature"],
            "regularization_weight": payload["regularization_weight"],
            "soar_weight": payload["soar_weight"],
            "reward_lr": payload["reward_lr"],
            "discriminator_lr": payload["discriminator_lr"],
            "policy_lr": payload["policy_lr"],
            "actor_lr": payload["actor_lr"],
            "target_update_interval": payload["target_update_interval"],
            "hidden_dims": payload["hidden_dims"],
            "soar_hidden_dims": payload["soar_hidden_dims"],
        }
        agent = cls.build_model(
            payload["observation_dim"],
            payload["action_dim"],
            "continuous",
            config,
        )
        agent.reward_network.load_state_dict(payload["reward_network_state_dict"])
        agent.discriminator.load_state_dict(payload["discriminator_state_dict"])
        agent.actor.load_state_dict(payload["actor_state_dict"])
        agent.q_network.load_state_dict(payload["q_network_state_dict"])
        agent.target_q_network.load_state_dict(payload["target_q_network_state_dict"])
        return agent

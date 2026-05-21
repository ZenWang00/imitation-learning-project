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
    csil_actor_loss,
    csil_continuous_critic_loss,
)
from src.imitation.soar import StateDiscriminator, soar_discriminator_loss


class ContinuousSOARCSILAgent(ImitationAgent):
    """SOAR-enhanced Coherent Soft Imitation Learning with continuous actions (Pendulum-v1).

    Extends ContinuousCSILAgent with a state occupancy discriminator (SOAR).
    Total reward: r(s,a) = r_coherent(s,a) + soar_weight * log_odds_D(s)
    """

    def __init__(
        self,
        bc_actor: BCContinuousActor,
        rl_actor: BCContinuousActor,
        discriminator: StateDiscriminator,
        q_network: ContinuousQNetwork,
        target_q_network: ContinuousQNetwork,
        *,
        gamma: float,
        temperature: float,
        soar_weight: float,
        bc_lr: float,
        discriminator_lr: float,
        policy_lr: float,
        actor_lr: float,
        target_update_interval: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.bc_actor = bc_actor.to(self.device)
        self.rl_actor = rl_actor.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.q_network = q_network.to(self.device)
        self.target_q_network = target_q_network.to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.gamma = gamma
        self.temperature = temperature
        self.soar_weight = soar_weight
        self.bc_lr = bc_lr
        self.discriminator_lr = discriminator_lr
        self.policy_lr = policy_lr
        self.actor_lr = actor_lr
        self.target_update_interval = target_update_interval
        self.bc_optimizer = torch.optim.Adam(self.bc_actor.parameters(), lr=bc_lr)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=discriminator_lr
        )
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=policy_lr)
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
        soar_hidden_dims = tuple(config.get("soar_hidden_dims", (128, 128)))
        return cls(
            bc_actor=BCContinuousActor(observation_dim, action_dim, hidden_dims),
            rl_actor=BCContinuousActor(observation_dim, action_dim, hidden_dims),
            discriminator=StateDiscriminator(observation_dim, soar_hidden_dims),
            q_network=ContinuousQNetwork(observation_dim, action_dim, hidden_dims),
            target_q_network=ContinuousQNetwork(observation_dim, action_dim, hidden_dims),
            gamma=float(config.get("gamma", 0.99)),
            temperature=float(config.get("temperature", 0.1)),
            soar_weight=float(config.get("soar_weight", 0.1)),
            bc_lr=float(config.get("bc_lr", 3e-4)),
            discriminator_lr=float(config.get("discriminator_lr", 3e-4)),
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

    def update_with_replay(
        self,
        expert_batch: TransitionBatch,
        replay_batch: TransitionBatch,
    ) -> dict[str, float]:
        # 1. Update SOAR discriminator
        disc_loss, disc_metrics = soar_discriminator_loss(
            expert_batch.observations,
            replay_batch.observations,
            self.discriminator,
        )
        self.discriminator_optimizer.zero_grad()
        disc_loss.backward()
        self.discriminator_optimizer.step()

        # 2. Compute total reward: coherent + SOAR bonus
        with torch.no_grad():
            coherent_rewards = coherent_reward_continuous(
                replay_batch.observations,
                replay_batch.actions,
                self.bc_actor,
                alpha=self.temperature,
            )
            soar_bonus = self.discriminator.log_odds(replay_batch.observations)
            total_rewards = coherent_rewards + self.soar_weight * soar_bonus

        # 3. Update critic
        q_loss, q_metrics = csil_continuous_critic_loss(
            replay_batch.observations,
            replay_batch.actions,
            replay_batch.next_observations,
            replay_batch.dones,
            total_rewards,
            self.q_network,
            self.target_q_network,
            self.rl_actor,
            gamma=self.gamma,
            temperature=self.temperature,
        )
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # 4. Update RL actor
        actor_loss, _ = csil_actor_loss(
            replay_batch.observations,
            self.q_network,
            self.rl_actor,
            temperature=self.temperature,
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        expert_rewards = coherent_reward_continuous(
            expert_batch.observations,
            expert_batch.actions,
            self.bc_actor,
            alpha=self.temperature,
        )
        return {
            "loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "expert_reward_mean": float(expert_rewards.mean().item()),
            "expert_term": 0.0,
            "replay_term": float(soar_bonus.mean().item()),
            "regularizer": 0.0,
            "q_mean": float(q_metrics["q_mean"].item()),
            "q_abs_max": float(q_metrics["q_mean"].abs().item()),
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
        torch.save(
            {
                "bc_actor_state_dict": self.bc_actor.state_dict(),
                "rl_actor_state_dict": self.rl_actor.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "q_network_state_dict": self.q_network.state_dict(),
                "target_q_network_state_dict": self.target_q_network.state_dict(),
                "gamma": self.gamma,
                "temperature": self.temperature,
                "soar_weight": self.soar_weight,
                "bc_lr": self.bc_lr,
                "discriminator_lr": self.discriminator_lr,
                "policy_lr": self.policy_lr,
                "actor_lr": self.actor_lr,
                "target_update_interval": self.target_update_interval,
                "observation_dim": self.bc_actor.backbone[0].in_features,
                "action_dim": self.bc_actor.action_dim,
                "hidden_dims": [
                    layer.out_features
                    for layer in self.bc_actor.backbone
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
            "soar_weight": payload["soar_weight"],
            "bc_lr": payload["bc_lr"],
            "discriminator_lr": payload["discriminator_lr"],
            "policy_lr": payload["policy_lr"],
            "actor_lr": payload["actor_lr"],
            "target_update_interval": payload["target_update_interval"],
            "hidden_dims": payload["hidden_dims"],
            "soar_hidden_dims": payload["soar_hidden_dims"],
        }
        agent = cls.build_model(
            payload["observation_dim"], payload["action_dim"], "continuous", config
        )
        agent.bc_actor.load_state_dict(payload["bc_actor_state_dict"])
        agent.bc_actor.eval()
        agent.bc_actor.requires_grad_(False)
        agent.rl_actor.load_state_dict(payload["rl_actor_state_dict"])
        agent.discriminator.load_state_dict(payload["discriminator_state_dict"])
        agent.q_network.load_state_dict(payload["q_network_state_dict"])
        agent.target_q_network.load_state_dict(payload["target_q_network_state_dict"])
        return agent

import torch

from src.imitation.iqlearn import (
    IQBatch,
    chi2_regularizer,
    inverse_soft_q_reward,
    iq_critic_loss,
    soft_policy_from_q,
    soft_value_from_q,
)


def test_soft_value_from_q_matches_manual_logsumexp() -> None:
    q_values = torch.tensor([[1.0, 2.0]])

    result = soft_value_from_q(q_values)

    expected = torch.log(torch.exp(torch.tensor(1.0)) + torch.exp(torch.tensor(2.0)))
    assert torch.allclose(result, expected.unsqueeze(0))


def test_inverse_soft_q_reward_masks_terminal_transition() -> None:
    q_values = torch.tensor([[2.0, 4.0]])
    next_q_values = torch.tensor([[10.0, 20.0]])
    actions = torch.tensor([1])
    dones = torch.tensor([1.0])

    result = inverse_soft_q_reward(
        q_values=q_values,
        actions=actions,
        next_q_values=next_q_values,
        dones=dones,
        gamma=0.99,
    )

    assert torch.allclose(result, torch.tensor([4.0]))


def test_iq_critic_loss_combines_expert_term_and_regularizer() -> None:
    class FixedQ(torch.nn.Module):
        def __init__(self, values: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("values", values)

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.values.expand(observations.shape[0], -1)

    expert_batch = IQBatch(
        observations=torch.zeros((2, 4)),
        actions=torch.tensor([0, 1]),
        next_observations=torch.zeros((2, 4)),
        dones=torch.ones(2),
    )
    replay_batch = IQBatch(
        observations=torch.zeros((2, 4)),
        actions=torch.tensor([0, 1]),
        next_observations=torch.zeros((2, 4)),
        dones=torch.ones(2),
    )
    q_network = FixedQ(torch.tensor([[2.0, 4.0]]))
    target_q_network = FixedQ(torch.tensor([[0.0, 0.0]]))

    loss, metrics = iq_critic_loss(
        expert_batch,
        replay_batch,
        q_network,
        target_q_network,
        regularization_weight=1.0,
    )

    rewards = torch.tensor([2.0, 4.0])
    replay_value = soft_value_from_q(torch.tensor([[2.0, 4.0], [2.0, 4.0]])).mean()
    expected = -rewards.mean() + replay_value + chi2_regularizer(rewards)
    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["expert_reward_mean"], rewards.mean())


def test_soft_policy_from_q_returns_distribution() -> None:
    q_values = torch.tensor([[1.0, 2.0, 3.0]])

    policy = soft_policy_from_q(q_values)

    assert torch.allclose(policy.sum(dim=-1), torch.ones(1))
    assert policy.argmax(dim=-1).item() == 2

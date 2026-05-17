from src.envs.specs import infer_env_spec


def test_infer_env_spec_for_discrete_env() -> None:
    spec = infer_env_spec("CartPole-v1")

    assert spec.observation_dim == 4
    assert spec.action_dim == 2
    assert spec.action_type == "discrete"


def test_infer_env_spec_for_continuous_env() -> None:
    spec = infer_env_spec("Pendulum-v1")

    assert spec.observation_dim == 3
    assert spec.action_dim == 1
    assert spec.action_type == "continuous"

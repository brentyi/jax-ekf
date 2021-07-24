import jaxlie
import numpy as onp
from jax import numpy as jnp

from ekf import (
    Array,
    EkfDefinition,
    ManifoldDefinition,
    MultivariateGaussian,
    euclidean_manifold,
)


def linear_system_example():
    """Kalman filter on a linear system."""

    state_dim: int = 5
    control_input_dim: int = 5
    observation_dim: int = 5

    euclidean_manifold.assert_validity(jnp.ones(5))

    A = onp.random.randn(state_dim, state_dim)
    B = onp.random.randn(state_dim, control_input_dim)
    C = onp.random.randn(observation_dim, state_dim)

    StateType = Array
    ObservationType = Array
    ControlInputType = Array
    ekf = EkfDefinition[StateType, ObservationType, ControlInputType](
        dynamics_model=lambda x, u: A @ x + B @ u,
        observation_model=lambda x: C @ x,
    )

    # In practice, we probably want to use an XLA loop primitive
    belief = MultivariateGaussian(mean=onp.ones(5), cov=onp.eye(5))
    for i in range(5):
        belief = ekf.predict(
            belief=belief,
            control_input=onp.zeros(5),
            dynamics_cov=onp.zeros((state_dim, state_dim)),
        )
        belief = ekf.correct(
            belief=belief,
            observation=MultivariateGaussian(
                mean=onp.random.randn(5),
                cov=onp.random.randn(observation_dim, observation_dim),
            ),
        )

    print(belief)


def SE3_system_example():
    """EKF, with states and observations on an SE(3) manifold."""

    point: jaxlie.MatrixLieGroup
    lie_group_manifold = ManifoldDefinition[jaxlie.MatrixLieGroup](
        boxplus=jaxlie.manifold.rplus,
        boxminus=jaxlie.manifold.rminus,
        local_dim_from_point=lambda point: point.tangent_dim,
    )
    lie_group_manifold.assert_validity(jaxlie.SE3.identity())

    StateType = jaxlie.SE3
    ObservationType = jaxlie.SE3
    ControlInputType = None

    ekf = EkfDefinition[StateType, ObservationType, ControlInputType](
        dynamics_model=lambda x, u: jaxlie.SE3.exp(
            onp.array([0.0, 0.1, 0.0, 0.0, 0.1, 0.0])
        )
        @ x,
        observation_model=lambda x: x,
        state_manifold=lie_group_manifold,
        observation_manifold=lie_group_manifold,
    )

    # In practice, we probably want to use an XLA loop primitive
    belief = MultivariateGaussian(jaxlie.SE3.identity(), cov=jnp.eye(6))
    for i in range(5):
        belief = ekf.predict(
            belief=belief, control_input=None, dynamics_cov=jnp.zeros((6, 6))
        )
        belief = ekf.correct(
            belief=belief,
            observation=MultivariateGaussian(jaxlie.SE3.identity(), cov=jnp.eye(6)),
        )
    print(belief)


if __name__ == "__main__":
    linear_system_example()
    SE3_system_example()

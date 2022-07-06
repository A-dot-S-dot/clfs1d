import numpy as np
from fem import FiniteElementSpace, GlobalFiniteElement
from fem.lagrange import LagrangeFiniteElementSpace
from math_types import FunctionRealDToRealD, FunctionRealToReal
from ode_solver.explicit_runge_kutta import (
    ExplicitRungeKuttaMethod,
    Heun,
    RungeKutta8,
    StrongStabilityPreservingRungeKutta3,
    StrongStabilityPreservingRungeKutta4,
)
from system.matrices.discrete_gradient import DiscreteGradientMatrix
from system.matrices.mass import MassMatrix
from system.vectors import DOFVectorBuilder
from system.vectors.averaged_gradient import AveragedGradientBuilder
from system.vectors.l2_projection_gradient import L2ProjectionGradientBuilder
from system.vectors.nonlinear_and_symmetric_stabilization import (
    NonlinearAndSymmetricStabilizationBuilder,
)
from system.vectors.nonlinear_and_vms_stabilization import (
    NonlinearAndVMSStabilizationBuilder,
)
from system.vectors.nonlinear_stabilization import NonlinearStabilizationBuilder
from system.vectors.symmetric_stabilization import SymmetricStabilizationBuilder
from system.vectors.vms_stabilization import VMSStabilizationBuilder

from .solver import HyperbolicPDESolver

ode_solver = {
    1: Heun,
    2: StrongStabilityPreservingRungeKutta3,
    3: StrongStabilityPreservingRungeKutta4,
    4: RungeKutta8,
    5: RungeKutta8,
    6: RungeKutta8,
    7: RungeKutta8,
}


class LinearAdvectionSolver(HyperbolicPDESolver):
    _element_space: FiniteElementSpace
    _ode_solver: ExplicitRungeKuttaMethod
    _mass: MassMatrix
    _discrete_gradient: DiscreteGradientMatrix
    _ode_right_hand_side_function: FunctionRealDToRealD
    _discrete_solution_dof_vector: np.ndarray
    _time: float

    def __init__(
        self,
        element_space: FiniteElementSpace,
        start_function: FunctionRealToReal,
        start_time=0,
    ):
        self._element_space = element_space
        self._mass = MassMatrix(element_space, build_inverse=True)
        self._discrete_gradient = DiscreteGradientMatrix(element_space)
        self._ode_right_hand_side_function = lambda dof_vector: self._mass.inverse(
            -self._discrete_gradient.dot(dof_vector) + self._stab()
        )
        self._discrete_solution_dof_vector = element_space.interpolate(start_function)
        self._time = start_time
        try:
            self._ode_solver = ode_solver[element_space.polynomial_degree](
                self._ode_right_hand_side_function,
                self._discrete_solution_dof_vector,
                start_time=self._time,
            )
        except KeyError:
            raise NotImplementedError(
                f"No optimal solver for p={element_space.polynomial_degree} available."
            )

    @property
    def discrete_solution(self) -> GlobalFiniteElement:
        return GlobalFiniteElement(
            self._element_space, self._discrete_solution_dof_vector
        )

    @property
    def time(self) -> float:
        return self._time

    def solve(self, target_time: float, time_steps_number: int):
        self._ode_solver.execute_steps(target_time, time_steps_number)
        self._discrete_solution_dof_vector = self._ode_solver.solution
        self._time = self._ode_solver.time

    def _stab(self) -> np.ndarray:
        raise NotImplementedError("must be implemented by subclasses")


class CGLinearAdvectionSolver(LinearAdvectionSolver):
    def _stab(self) -> float:
        return 0


class StabilizedLinearAdvectionSolver(LinearAdvectionSolver):
    _stabilization_builder: DOFVectorBuilder
    _stabilization_parameter: float

    def _get_gradient_approximation_builder(
        self, gradient_approximation: str
    ) -> DOFVectorBuilder:
        if gradient_approximation == "nodal":
            return AveragedGradientBuilder(self._element_space)
        elif gradient_approximation == "l2_projection":
            return L2ProjectionGradientBuilder(self._element_space, self._mass)
        else:
            raise NotImplementedError

    @property
    def stabilization_parameter(self) -> float:
        return self._stabilization_parameter

    def _stab(self) -> np.ndarray:
        return -self._stabilization_builder.build_vector(
            self._discrete_solution_dof_vector
        )


class VMSStabilizedLinearAdvectionSolver(StabilizedLinearAdvectionSolver):
    def __init__(
        self,
        element_space: LagrangeFiniteElementSpace,
        start_function: FunctionRealToReal,
        stabilization_parameter=0.5,
        gradient_approximation="nodal",
    ):
        LinearAdvectionSolver.__init__(self, element_space, start_function)

        self._stabilization_parameter = stabilization_parameter

        gradient_approximation_builder = self._get_gradient_approximation_builder(
            gradient_approximation
        )
        self._stabilization_builder = VMSStabilizationBuilder(
            element_space,
            stabilization_parameter,
            gradient_approximation_builder,
        )


class SymmetricStabilizedLinearAdvectionSolver(StabilizedLinearAdvectionSolver):
    def __init__(
        self,
        element_space: LagrangeFiniteElementSpace,
        start_function: FunctionRealToReal,
        stabilization_parameter=0.5,
        gradient_approximation="nodal",
    ):
        LinearAdvectionSolver.__init__(self, element_space, start_function)
        self._stabilization_parameter = stabilization_parameter

        gradient_approximation_builder = self._get_gradient_approximation_builder(
            gradient_approximation
        )
        self._stabilization_builder = SymmetricStabilizationBuilder(
            element_space,
            stabilization_parameter,
            gradient_approximation_builder,
        )


class NonlinearStabilizedLinearAdvectionSolver(StabilizedLinearAdvectionSolver):
    def __init__(
        self,
        element_space: LagrangeFiniteElementSpace,
        start_function: FunctionRealToReal,
        stabilization_parameter=0.5,
        gradient_approximation="nodal",
    ):
        LinearAdvectionSolver.__init__(self, element_space, start_function)
        self._stabilization_parameter = stabilization_parameter

        gradient_approximation_builder = self._get_gradient_approximation_builder(
            gradient_approximation
        )
        self._stabilization_builder = NonlinearStabilizationBuilder(
            element_space,
            stabilization_parameter,
            gradient_approximation_builder,
        )


class NonlinearAndVMSStabilizedLinearAdvectionSolver(StabilizedLinearAdvectionSolver):
    def __init__(
        self,
        element_space: LagrangeFiniteElementSpace,
        start_function: FunctionRealToReal,
        stabilization_parameter=0.5,
        gradient_approximation="nodal",
    ):
        LinearAdvectionSolver.__init__(self, element_space, start_function)
        self._stabilization_parameter = stabilization_parameter

        gradient_approximation_builder = self._get_gradient_approximation_builder(
            gradient_approximation
        )
        self._stabilization_builder = NonlinearAndVMSStabilizationBuilder(
            element_space,
            stabilization_parameter,
            gradient_approximation_builder,
        )


class NonlinearAndSymmetricStabilizedLinearAdvectionSolver(
    StabilizedLinearAdvectionSolver
):
    def __init__(
        self,
        element_space: LagrangeFiniteElementSpace,
        start_function: FunctionRealToReal,
        stabilization_parameter=0.5,
        gradient_approximation="nodal",
    ):
        LinearAdvectionSolver.__init__(self, element_space, start_function)
        self._stabilization_parameter = stabilization_parameter

        gradient_approximation_builder = self._get_gradient_approximation_builder(
            gradient_approximation
        )
        self._stabilization_builder = NonlinearAndSymmetricStabilizationBuilder(
            element_space,
            stabilization_parameter,
            gradient_approximation_builder,
        )

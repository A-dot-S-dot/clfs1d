from abc import ABC, abstractmethod
from argparse import Namespace

import defaults
from fem.lagrange import LagrangeFiniteElementSpace
from math_types import FunctionRealToReal
from mesh import Interval, UniformMesh
from pde_solver.linear_advection import (
    CGLinearAdvectionSolver,
    LinearAdvectionSolver,
    NonlinearAndSymmetricStabilizedLinearAdvectionSolver,
    NonlinearAndVMSStabilizedLinearAdvectionSolver,
    NonlinearStabilizedLinearAdvectionSolver,
    SymmetricStabilizedLinearAdvectionSolver,
    VMSStabilizedLinearAdvectionSolver,
)


class SolverTask(ABC):
    _args: Namespace
    _mesh: UniformMesh
    _element_space: LagrangeFiniteElementSpace
    _solver: LinearAdvectionSolver
    _exact_solution: FunctionRealToReal

    def __init__(self, args: Namespace):
        self._args = args

        self._build_mesh()
        self._build_element_space()
        self._build_solver()
        self._calculate_solution()

        self._build_exact_solution()

    def _build_mesh(self):
        domain = Interval(defaults.a, defaults.b)
        self._mesh = UniformMesh(domain, self._args.elementsNumber)

    def _build_element_space(self):
        self._element_space = LagrangeFiniteElementSpace(
            self._mesh, self._args.polynomialDegree
        )

    def _build_solver(self):
        if self._args.solver == "cg":
            self._solver = CGLinearAdvectionSolver(
                self._element_space, defaults.initial_data
            )
        elif self._args.solver == "vms":
            self._solver = VMSStabilizedLinearAdvectionSolver(
                self._element_space,
                defaults.initial_data,
                stabilization_parameter=self._args.omega,
                gradient_approximation=self._args.gradientApproximation,
            )
        elif self._args.solver == "sym":
            self._solver = SymmetricStabilizedLinearAdvectionSolver(
                self._element_space,
                defaults.initial_data,
                stabilization_parameter=self._args.omega,
                gradient_approximation=self._args.gradientApproximation,
            )
        elif self._args.solver == "nonlin":
            self._solver = NonlinearStabilizedLinearAdvectionSolver(
                self._element_space,
                defaults.initial_data,
                stabilization_parameter=self._args.omega,
                gradient_approximation=self._args.gradientApproximation,
            )
        elif self._args.solver == "nvms":
            self._solver = NonlinearAndVMSStabilizedLinearAdvectionSolver(
                self._element_space,
                defaults.initial_data,
                stabilization_parameter=self._args.omega,
                gradient_approximation=self._args.gradientApproximation,
            )
        elif self._args.solver == "nsym":
            self._solver = NonlinearAndSymmetricStabilizedLinearAdvectionSolver(
                self._element_space,
                defaults.initial_data,
                stabilization_parameter=self._args.omega,
                gradient_approximation=self._args.gradientApproximation,
            )
        else:
            raise NotImplementedError

    def _calculate_solution(self):
        target_time = self._args.time
        time_steps_number = len(self._mesh) * self._args.courantFactor
        self._solver.solve(target_time, time_steps_number)

    def _build_exact_solution(self):
        time = self._args.time
        self._exact_solution = lambda x: defaults.exact_solution(x, time)

    @abstractmethod
    def execute(self):
        ...

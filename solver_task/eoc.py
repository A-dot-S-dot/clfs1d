"""This module provides a solver task for analyzing error between an exact
solution and a discrete one of a PDE with diffrent norms.

"""
import time
from argparse import Namespace
from typing import List

import numpy as np
from math_types import FunctionRealToReal
from mesh import UniformMesh
from pde_solver.linear_advection import (
    CGLinearAdvectionSolver,
    NonlinearAndSymmetricStabilizedLinearAdvectionSolver,
    NonlinearAndVMSStabilizedLinearAdvectionSolver,
    NonlinearStabilizedLinearAdvectionSolver,
    StabilizedLinearAdvectionSolver,
    SymmetricStabilizedLinearAdvectionSolver,
    VMSStabilizedLinearAdvectionSolver,
)
from quadrature.norm import L1Norm, L2Norm, LInfinityNorm, Norm

from .task import SolverTask


class EOCMessenger:
    _norm: Norm
    _norm_description: str
    _error = None
    _eoc = None

    def __init__(self, norm: Norm, norm_description: str):
        self._norm = norm
        self._norm_description = norm_description

    @property
    def message(self) -> str:
        message = f"{self._norm_description}:\t error={self._error:.3E}"
        if self._eoc is not None:
            message += f", eoc={self._eoc:.2f}"

        return message

    def calculate_error(
        self, exact_solution: FunctionRealToReal, discrete_solution: FunctionRealToReal
    ):
        function = lambda x: discrete_solution(x) - exact_solution(x)
        new_error = self._norm(function)

        if self._error is None:
            self._error = new_error
        else:
            self._eoc = np.log2(self._error / new_error)
            self._error = new_error

    def update_norm(self, mesh: UniformMesh):
        self._norm.set_mesh(mesh)


class EOCTask(SolverTask):
    _eoc_runs_number: int
    _eoc_messenger: List[EOCMessenger]
    _calculation_time = None

    def __init__(self, args: Namespace):
        SolverTask.__init__(self, args)
        self._build_eoc_messenger()
        self._build_eoc_runs()

    def _build_eoc_messenger(self):
        l2_norm = L2Norm(self._mesh, self._element_space.polynomial_degree + 1)
        l1_norm = L1Norm(self._mesh, self._element_space.polynomial_degree)
        l_infinity_norm = LInfinityNorm(
            self._mesh, self._element_space.polynomial_degree + 5
        )

        self._eoc_messenger = [
            EOCMessenger(l2_norm, "L2"),
            EOCMessenger(l1_norm, "L1"),
            EOCMessenger(l_infinity_norm, "Linf"),
        ]

    def _build_eoc_runs(self):
        self._eoc_runs_number = self._args.refine

    def execute(self):
        self._print_general_info()

        self._calculate_errors()
        self._print_eoc_info()

        for _ in range(self._eoc_runs_number):
            self._refine()
            self._calculate_errors()
            self._print_eoc_info()

    def _print_general_info(self):
        message = f"p={self._element_space.polynomial_degree}, eoc_runs={self._eoc_runs_number}"
        if isinstance(self._solver, CGLinearAdvectionSolver):
            message = "CG, " + message
        elif isinstance(self._solver, StabilizedLinearAdvectionSolver):
            message = (
                message
                + f", omega={self._solver.stabilization_parameter}, gradient_approx.={self._args.gradientApproximation}"
            )
            if isinstance(self._solver, VMSStabilizedLinearAdvectionSolver):
                message = "VMS, " + message
            elif isinstance(self._solver, SymmetricStabilizedLinearAdvectionSolver):
                message = "Sym. Stab., " + message
            elif isinstance(self._solver, NonlinearStabilizedLinearAdvectionSolver):
                message = "Nonlin. Stab., " + message
            elif isinstance(
                self._solver, NonlinearAndVMSStabilizedLinearAdvectionSolver
            ):
                message = "Nonlin. & VMS Stab., " + message
            elif isinstance(
                self._solver, NonlinearAndSymmetricStabilizedLinearAdvectionSolver
            ):
                message = "Nonlin. & Sym. Stab., " + message
        else:
            raise NotImplementedError

        print(message)

    def _calculate_errors(self):
        for eoc_messenger in self._eoc_messenger:
            eoc_messenger.calculate_error(
                self._exact_solution, self._solver.discrete_solution
            )

    def _print_eoc_info(self):
        message = f"Nh={self._element_space.dimension}"
        if self._calculation_time is not None:
            message += f", duration={self._calculation_time:.2f}s"

        for eoc_messenger in self._eoc_messenger:
            message += "\n" + eoc_messenger.message

        print(message + "\n")

    def _refine(self):
        self._mesh = self._mesh.refine()
        self._build_element_space()
        self._build_solver()
        self._update_eoc_messenger()

        start = time.time()
        self._calculate_solution()
        self._calculation_time = time.time() - start

    def _update_eoc_messenger(self):
        for eoc_messenger in self._eoc_messenger:
            eoc_messenger.update_norm(self._mesh)

from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from math_types import FunctionRealToReal
from mesh import Interval
from pde_solver.linear_advection import (
    CGLinearAdvectionSolver,
    LinearAdvectionSolver,
    NonlinearAndSymmetricStabilizedLinearAdvectionSolver,
    NonlinearAndVMSStabilizedLinearAdvectionSolver,
    NonlinearStabilizedLinearAdvectionSolver,
    StabilizedLinearAdvectionSolver,
    SymmetricStabilizedLinearAdvectionSolver,
    VMSStabilizedLinearAdvectionSolver,
)

from .task import SolverTask


class FunctionPlotter:
    _grid: np.ndarray
    _title: str
    _suptitle: str

    def __init__(self, interval: Interval):
        self._grid = np.linspace(interval.a, interval.b)

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, title: str):
        self._title = title
        plt.title(title)

    @property
    def suptitle(self) -> str:
        return self._suptitle

    @suptitle.setter
    def suptitle(self, suptitle: str):
        self._suptitle = suptitle
        plt.suptitle(suptitle, fontsize=14, fontweight="bold")

    def add_function(self, function: FunctionRealToReal, function_label: str):
        function_values = np.array([function(x) for x in self._grid])

        plt.plot(self._grid, function_values, label=function_label)

    def save(self, path="output/plot.png"):
        self._setup()
        plt.savefig(path)

    def show(self):
        self._setup()
        plt.show()
        plt.close()

    def _setup(self):
        plt.xlabel("x")
        plt.legend()


class SolutionPlotter(FunctionPlotter):
    _solver: LinearAdvectionSolver

    def __init__(self, interval: Interval, solver: LinearAdvectionSolver):
        FunctionPlotter.__init__(self, interval)

        self._solver = solver
        self._build_suptitle()

    def _build_suptitle(self):
        if isinstance(self._solver, CGLinearAdvectionSolver):
            self.suptitle = "Discrete Solution with CG"
        elif isinstance(self._solver, VMSStabilizedLinearAdvectionSolver):
            self.suptitle = "Discrete Solution with VMS stabilization"
        elif isinstance(self._solver, SymmetricStabilizedLinearAdvectionSolver):
            self.suptitle = "Discrete Solution with symmetric stabilization"
        elif isinstance(self._solver, NonlinearStabilizedLinearAdvectionSolver):
            self.suptitle = "Discrete Solution with nonlinear stabilization"
        elif isinstance(self._solver, NonlinearAndVMSStabilizedLinearAdvectionSolver):
            self.suptitle = "Discrete Solution with nonlin. and VMS stabilization"
        elif isinstance(
            self._solver, NonlinearAndSymmetricStabilizedLinearAdvectionSolver
        ):
            self.suptitle = "Discrete Solution with nonlin. and sym. stabilization"
        else:
            raise NotImplementedError("Suptitle is not implemented yet.")

    def build_title(
        self,
        polynomial_degree: int,
        step_length: float,
        courant_factor: float,
        gradient_approximation=None,
    ):
        self.title = f"$p={polynomial_degree},\, h={step_length:.2f},\, \Delta t={1/courant_factor:.1f}*h"
        if isinstance(self._solver, StabilizedLinearAdvectionSolver):
            self.title += f",\, \omega={{{self._solver.stabilization_parameter:.2f}}},\, g=${gradient_approximation}"
        else:
            self.title += "$"


class PlotTask(SolverTask):
    _plotter: SolutionPlotter
    _args: Namespace

    def __init__(self, args: Namespace):
        SolverTask.__init__(self, args)

        domain = self._mesh.domain
        self._plotter = SolutionPlotter(domain, self._solver)

        self._add_functions()
        self._add_title()

    def _add_functions(self):
        self._plotter.add_function(
            self._solver.discrete_solution, f"$u_h(\cdot, {self._solver.time:.2f})$"
        )
        self._plotter.add_function(
            self._exact_solution, f"$u(\cdot, {self._args.time:.2f})$"
        )

    def _add_title(self):
        step_length = 1 / self._args.elementsNumber
        self._plotter.build_title(
            self._args.polynomialDegree,
            step_length,
            self._args.courantFactor,
            self._args.gradientApproximation,
        )

    def execute(self):
        if self._args.save:
            self._plotter.save()
        else:
            self._plotter.show()

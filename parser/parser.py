from argparse import ArgumentParser, Namespace
from typing import Any, List

import defaults
from . import types as parser_type


class ArgumentParserFEM1D:
    """Parser for command line arguments."""

    _parser = ArgumentParser(
        description="""This program executes diffrent task for different
        solvers. """,
    )
    _current_parser_layer: List[Any] = [_parser]

    def __init__(self):
        self._add_parser_layer(self._add_problem_parsers)
        self._add_parser_layer(self._add_solver_parsers)
        self._add_arguments_to_current_layer(self._add_mesh_arguments)
        self._add_arguments_to_current_layer(self._add_element_space_arguments)
        self._add_parser_layer(self._add_task_parsers)
        self._add_arguments_to_current_layer(self._add_profile_argument)

    def _add_parser_layer(self, add_parsers_function):
        current_parsers = self._current_parser_layer
        self._clear_current_parser_layer()

        for parser in current_parsers:
            current_parsers = add_parsers_function(parser)
            self._add_to_current_parser_layer(current_parsers)

    def _clear_current_parser_layer(self):
        self._current_parser_layer = []

    def _add_to_current_parser_layer(self, parsers):
        self._current_parser_layer.extend(parsers._name_parser_map.values())

    def _add_arguments_to_current_layer(self, add_arguments_function):
        for parser in self._current_parser_layer:
            add_arguments_function(parser)

    def _add_solver_parsers(self, parser):
        solver_parsers = parser.add_subparsers(
            title="solver",
            help="available solver for the transport problem using diffrent stabilization techniques",
            dest="solver",
            required=True,
        )

        self._add_cg_parser(solver_parsers)
        self._add_vms_parser(solver_parsers)
        self._add_symmetric_stabilization_parser(solver_parsers)
        self._add_nonlinear_parser(solver_parsers)
        self._add_nonlinear_and_vms_stabilization_parser(solver_parsers)
        self._add_nonlinear_and_symmetric_stabilization_parser(solver_parsers)

        return solver_parsers

    def _add_cg_parser(self, solver_parsers):
        solver_parsers.add_parser(
            "cg",
            help="classical CG (continuous Galerkin) solver",
            conflict_handler="resolve",
        )

    def _add_vms_parser(self, solver_parsers):
        vms_parser = solver_parsers.add_parser(
            "vms",
            help="solver using VMS stabilization technique",
            conflict_handler="resolve",
        )
        vms_parser.add_argument(
            "-o",
            "--omega",
            help="specify stabilization factor",
            type=parser_type.percent_number,
            default=defaults.stabilization_parameter,
        )

        vms_parser.add_argument(
            "--gradientApproximation",
            help="select used gradient approximation for stabilization",
            choices=["nodal", "l2_projection"],
            default=defaults.stabilization_gradient_approximation,
        )

    def _add_nonlinear_parser(self, solver_parsers):
        nonlinear_parser = solver_parsers.add_parser(
            "nonlin",
            help="solver using nonlinear stabilization technique",
            conflict_handler="resolve",
        )
        nonlinear_parser.add_argument(
            "-o",
            "--omega",
            help="specify stabilization parameter",
            type=parser_type.percent_number,
            default=defaults.stabilization_parameter,
        )
        nonlinear_parser.add_argument(
            "--gradientApproximation",
            help="select used gradient approximation for stabilization",
            choices=["nodal", "l2_projection"],
            default=defaults.stabilization_gradient_approximation,
        )

    def _add_symmetric_stabilization_parser(self, solver_parsers):
        symmetric_stabilization_parser = solver_parsers.add_parser(
            "sym",
            help="solver using symmetric stabilization technique",
            conflict_handler="resolve",
        )
        symmetric_stabilization_parser.add_argument(
            "-o",
            "--omega",
            help="specify stabilization parameter",
            type=parser_type.percent_number,
            default=defaults.stabilization_parameter,
        )
        symmetric_stabilization_parser.add_argument(
            "--gradientApproximation",
            help="select used gradient approximation for stabilization",
            choices=["nodal", "l2_projection"],
            default=defaults.stabilization_gradient_approximation,
        )

    def _add_nonlinear_and_vms_stabilization_parser(self, solver_parsers):
        nonlinear_and_vms_stabilization_parser = solver_parsers.add_parser(
            "nvms",
            help="solver using nonlinear and vms stabilization techniques",
            conflict_handler="resolve",
        )
        nonlinear_and_vms_stabilization_parser.add_argument(
            "-o",
            "--omega",
            help="specify stabilization parameter",
            type=parser_type.percent_number,
            default=defaults.stabilization_parameter,
        )
        nonlinear_and_vms_stabilization_parser.add_argument(
            "--gradientApproximation",
            help="select used gradient approximation for stabilization",
            choices=["nodal", "l2_projection"],
            default=defaults.stabilization_gradient_approximation,
        )

    def _add_nonlinear_and_symmetric_stabilization_parser(self, solver_parsers):
        nonlinear_and_symmetric_stabilization_parser = solver_parsers.add_parser(
            "nsym",
            help="solver using nonlinear and symmetric stabilization techniques",
            conflict_handler="resolve",
        )
        nonlinear_and_symmetric_stabilization_parser.add_argument(
            "-o",
            "--omega",
            help="specify stabilization parameter",
            type=parser_type.percent_number,
            default=defaults.stabilization_parameter,
        )
        nonlinear_and_symmetric_stabilization_parser.add_argument(
            "--gradientApproximation",
            help="select used gradient approximation for stabilization",
            choices=["nodal", "l2_projection"],
            default=defaults.stabilization_gradient_approximation,
        )

    def _add_problem_parsers(self, parser):
        problem_parsers = parser.add_subparsers(
            title="problem",
            help="select problem to set defaults",
            dest="problem",
            required=True,
        )

        self._add_advection_parser(problem_parsers)

        return problem_parsers

    def _add_advection_parser(self, problem_parsers):
        advection_parser = problem_parsers.add_parser(
            "advection", help="linear advection: ut+ux=0"
        )
        advection_parser.add_argument(
            "problemNumber", help="0: rectangle; 1: gaussian bell; 2: cosine", type=int
        )

    def _add_task_parsers(self, parser):
        task_parsers = parser.add_subparsers(
            title="solver tasks",
            help="available tasks for the solver",
            dest="solver_task",
            required=True,
        )

        self._add_plot_parser(task_parsers)
        self._add_eoc_parser(task_parsers)

        return task_parsers

    def _add_plot_parser(self, task_parsers):
        plot_parser = task_parsers.add_parser(
            "plot",
            help="plot solution at certain time",
        )
        plot_parser.add_argument(
            "--save", help="save plot without showing it", action="store_true"
        )

    def _add_eoc_parser(self, task_parsers):
        eoc_parser = task_parsers.add_parser(
            "eoc",
            help="execute eoc test",
        )
        eoc_parser.add_argument(
            "--refine",
            help="specify how many times the mesh should be refined",
            type=parser_type.positive_int,
            default=defaults.refine_number,
        )

    def _add_mesh_arguments(self, parser):
        mesh_argument_group = parser.add_argument_group("mesh arguments")
        mesh_argument_group.add_argument(
            "--elementsNumber",
            help="specify simplex number of the created mesh",
            type=parser_type.positive_int,
            default=defaults.elements_number,
        )
        mesh_argument_group.add_argument(
            "-T",
            "--time",
            help="specify end time for calculation of the discrete solution",
            type=parser_type.positive_float,
            default=defaults.T,
        )
        mesh_argument_group.add_argument(
            "--courantFactor",
            help="specify the factor for the number of time steps depending on the number of simplices in the used mesh",
            type=parser_type.positive_int,
            default=defaults.courant_factor,
        )

    def _add_element_space_arguments(self, parser):
        element_space_argument_group = parser.add_argument_group(
            "finite element space arguments"
        )
        element_space_argument_group.add_argument(
            "-p",
            "--polynomialDegree",
            help="specify polynomial degree of the element space",
            type=parser_type.positive_int,
            default=defaults.polynomial_degree,
        )

    def _add_profile_argument(self, parser):
        parser.add_argument(
            "--profile",
            help="profile program for optimization purposes",
            action="store_true",
        )

    def parse_args(self) -> Namespace:
        return self._parser.parse_args()

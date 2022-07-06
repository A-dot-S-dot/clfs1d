import defaults

from numpy import exp, cos, pi
from math_types import FunctionRealToReal


def build_defaults(problem: str, problem_number: int):
    if problem == "advection":
        if problem_number == 0:
            defaults.initial_data = lambda x: float(x >= 0.2 and x <= 0.4)
        elif problem_number == 1:
            defaults.initial_data = (lambda x: exp(-100 * (x - 0.5) ** 2),)
        elif problem_number == 2:
            defaults.initial_data = lambda x: cos(2 * pi * (x - 0.5))
        else:
            raise ValueError(f"problem number {problem_number} does not exist")

        defaults.exact_solution = (
            lambda x, t: advection_periodic_boundaries_exact_solution(
                x, t, defaults.initial_data
            )
        )

    else:
        raise ValueError(f"problem {problem} is not implemented")


def advection_periodic_boundaries_exact_solution(
    x: float, t: float, initial_data: FunctionRealToReal
):
    arg = (x - t) % (defaults.b - defaults.a)
    return initial_data(arg)

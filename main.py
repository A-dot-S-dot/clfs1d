"""This is the main script for Solving linear advection.

For setting up parameters use the file `defauls.py` or options.
More information are available with "python3 main.py --help".

"""

from parser import ArgumentParserFEM1D

import profiler
from solver_task import PlotTask, EOCTask

from defaults_builder import build_defaults


@profiler.profile
def main() -> None:
    parser = ArgumentParserFEM1D()
    args = parser.parse_args()

    build_defaults(args.problem, args.problemNumber)

    profiler.PRINT_PROFILE = args.profile

    if args.solver_task == "plot":
        task = PlotTask(args)
    elif args.solver_task == "eoc":
        task = EOCTask(args)
    else:
        raise NotImplementedError

    task.execute()


if __name__ == "__main__":
    main()

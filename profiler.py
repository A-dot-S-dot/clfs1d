"""This module provides profile functionalities.

If a function is decorated with `profile` but no profile information should be
printed, please set `print_profile` to `False`.

"""
import cProfile, pstats, io

PRINT_PROFILE = True


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        fnc(*args, **kwargs)
        pr.disable()

        if PRINT_PROFILE:
            s = io.StringIO()
            sortby = "cumulative"
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print("\n", s.getvalue())

    return inner

from dataclasses import dataclass
from typing import Optional
from src.algorithms.solution import Solution

@dataclass
class RunResult:
    """One execution of an algorithm.

    Returned per call so a reused algorithm never carries state from a previous run.

    `proven` answers whether the fitness is a *demonstrated* optimum, not whether it
    happens to be one -- a solver that ran out of time may well have found the best
    solution without ever proving it. None for algorithms that cannot tell either way.
    """
    solution: Solution
    proven: Optional[bool] = None
    iterations: Optional[int] = None

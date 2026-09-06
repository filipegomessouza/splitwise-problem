from dataclasses import dataclass
from src.algorithms.solution import Solution

@dataclass
class RunResult:
    """One execution of an algorithm.

    Returned per call so a reused algorithm never carries state from a previous run.
    """
    solution: Solution

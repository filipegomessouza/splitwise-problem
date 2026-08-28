from dataclasses import dataclass
from typing import Optional
from src.algorithms.solution import Solution

@dataclass
class RunResult:
    """One execution of an algorithm: the solution plus whatever the solver reported.

    Returned per call so a reused algorithm never carries state from a previous run.
    """
    solution: Solution
    status: str
    gap: Optional[float] = None

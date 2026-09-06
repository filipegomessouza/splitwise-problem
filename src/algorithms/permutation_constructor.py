from typing import Protocol, Tuple
import numpy as np
from src.algorithms.solution import Solution
from src.instance.instance import Instance

class PermutationConstructor(Protocol):
    """What a local search needs from whoever hands it a starting point.

    Structural on purpose: a constructor is already a BaseAlgorithm, and making it inherit
    a second base only to be accepted here would buy nothing. Anything that can produce an
    order and the solution that order decodes to qualifies.
    """

    def construct(self, instance: Instance) -> Tuple[np.ndarray, Solution]:
        """An order of the instance's survivors, and the solution it decodes to."""
        ...

    def supports(self, instance: Instance) -> bool:
        ...

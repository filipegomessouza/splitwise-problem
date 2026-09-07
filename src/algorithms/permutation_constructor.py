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
        """An order of the instance's survivors, and a solution to start from.

        A constructor that works by decoding an order returns the decode of this one. One
        that does not -- the greedy settles every survivor in a single pass -- returns its
        own solution instead, and then the pair does not correspond.
        """
        ...

    def supports(self, instance: Instance) -> bool:
        ...

    def name(self) -> str:
        """Short identifier, which a search built on this constructor names itself after."""
        ...

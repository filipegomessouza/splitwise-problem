from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from src.algorithms.base_algorithm import METRICS, BaseAlgorithm
from src.algorithms.permutation_constructor import PermutationConstructor
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.instance.instance import Instance

class ImprovementAlgorithm(BaseAlgorithm, ABC):
    """Improves the starting point an injected constructor hands it.

    What a search does to a solution is the subclass's business, through improve. What
    every search shares is how it sits in a report and in a chain: it is named after its
    constructor, reports its own iterations, and is itself a PermutationConstructor, so
    that another search can start from where this one stopped --
    MergeSearchAlgorithm(FirstImprovementAlgorithm(BrkgaAlgorithm())) runs all three.

    The constructor is optional, because improve never reads it: a search used only for
    its improve -- a neighbourhood inside VndAlgorithm -- needs none. run and construct
    are what need one, and say so.
    """

    def __init__(self, constructor: Optional[PermutationConstructor] = None) -> None:
        self._constructor = constructor

    @property
    def constructor(self) -> Optional[PermutationConstructor]:
        return self._constructor

    def set_constructor(self, constructor: Optional[PermutationConstructor]) -> None:
        """Point run at another starting point, without rebuilding the search."""
        self._constructor = constructor

    def name(self) -> str:
        """Named after the constructor, because the constructor is half the algorithm.

        Two searches of the same strategy over different constructors are different
        algorithms and need different columns; a fixed name would have them collide in a
        report, one quietly overwriting the other. Without one, the suffix alone.
        """
        if self._constructor is None:
            return self.suffix()

        return f"{self._constructor.name()}_{self.suffix()}"

    @abstractmethod
    def suffix(self) -> str:
        """Short tag for this strategy, appended to the constructor's name."""
        pass

    def supports(self, instance: Instance) -> bool:
        # without a constructor, let run fail loudly: unsupported would skip it in silence
        return self._constructor is None or self._constructor.supports(instance)

    def metrics(self) -> List[str]:
        return METRICS + ['iterations']

    def run(self, instance: Instance) -> RunResult:
        order, solution = self._starting_point(instance)
        _, iterations, improved = self.improve(order, solution)

        return RunResult(solution=improved, iterations=iterations)

    def construct(self, instance: Instance) -> Tuple[np.ndarray, Solution]:
        """The constructor's starting point, improved: what lets searches be chained."""
        order, solution = self._starting_point(instance)
        improved_order, _, improved = self.improve(order, solution)

        return improved_order, improved

    def _starting_point(self, instance: Instance) -> Tuple[np.ndarray, Solution]:
        if self._constructor is None:
            raise ValueError(f"{self.name()} needs a constructor to run on its own")

        return self._constructor.construct(instance)

    @abstractmethod
    def improve(self, order: np.ndarray, solution: Solution) -> Tuple[np.ndarray, int, Solution]:
        """Refine an order and its incumbent; the order, the iterations, and the solution.

        Never returns a solution worse than `solution`. The order returned decodes to the
        solution returned whenever the search moved.
        """
        pass

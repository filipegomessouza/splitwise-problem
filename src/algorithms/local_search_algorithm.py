from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from src.algorithms.base_algorithm import METRICS, BaseAlgorithm
from src.algorithms.order_evaluator import OrderEvaluator
from src.algorithms.permutation_constructor import PermutationConstructor
from src.algorithms.permutation_decoder import PermutationDecoder
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.instance.instance import Instance

# the whole neighbourhood is O(n^2) orders and scoring one is O(n), so the work per
# iteration grows as n^3. The evaluator, and then compiling it, each took the constant down
# by about an order of magnitude -- never the exponent
MAX_PEOPLE = None

class LocalSearchAlgorithm(BaseAlgorithm, ABC):
    """Improves an order by swapping two people at a time, until nothing gets better.

    The neighbourhood is every swap of positions (i, j), i < j: O(n^2) orders, each scored
    by the scan that decoding it would run. Which neighbour to accept is the one thing
    subclasses decide -- best-improvement scans it all and keeps the best, first-improvement
    stops at the first gain. Both run inside one compiled loop, which reads the choice as a
    flag and checks it only when a neighbour improves, so the O(n^2) inner loop never
    dispatches on it.

    A search improves an order; it does not produce one. run only exists so the search can
    take its turn in a report, and it gets its starting point from an injected constructor.
    improve, which is the real entry point, ignores the constructor entirely: one object
    can refine any number of starting points, whatever produced them.
    """

    def __init__(
        self,
        constructor: PermutationConstructor,
        max_iterations: Optional[int] = None,
    ) -> None:
        self._constructor = constructor
        self._max_iterations = max_iterations
        self._decoder = PermutationDecoder()

    @property
    def constructor(self) -> PermutationConstructor:
        return self._constructor

    def set_constructor(self, constructor: PermutationConstructor) -> None:
        """Point run at another starting point, without rebuilding the search."""
        self._constructor = constructor

    def name(self) -> str:
        """Named after the constructor, because the constructor is half the algorithm.

        Two searches of the same strategy over different constructors are different
        algorithms and need different columns; a fixed name would have them collide in a
        report, one quietly overwriting the other.
        """
        return f"{self._constructor.name()}_{self.suffix()}"

    @abstractmethod
    def suffix(self) -> str:
        """Short tag for this strategy, appended to the constructor's name."""
        pass

    def supports(self, instance: Instance) -> bool:
        return (
            MAX_PEOPLE is None or len(instance.balances) <= MAX_PEOPLE
            and self._constructor.supports(instance)
        )

    def metrics(self) -> List[str]:
        return METRICS + ['iterations']

    def run(self, instance: Instance) -> RunResult:
        order, solution = self._constructor.construct(instance)
        _, iterations, improved = self.improve(order, solution)

        return RunResult(solution=improved, iterations=iterations)

    def improve(self, order: np.ndarray, solution: Solution) -> Tuple[np.ndarray, int, Solution]:
        """Refine an order until no swap improves it, or max_iterations run out.

        `solution` is the incumbent, and its fitness is the bar every neighbour has to
        clear. Passing it in rather than decoding it here is what keeps a constructor's
        work from being repeated.

        It need not be what `order` decodes to. A constructor that does not work by
        decoding -- the greedy -- hands over its own solution, which no order reproduces,
        and then the bar comes from outside the space being searched. That is deliberate:
        it makes the search unable to report anything worse than its constructor. What it
        costs is one guarantee, so what comes back is worth stating exactly:

        - the fitness returned is never above the fitness handed in;
        - the order returned decodes to the solution returned if and only if the search
          moved. With no move the pair comes back exactly as it arrived, mismatch and all.

        The order comes back alongside the solution because a metaheuristic needs it: the
        refined order is the part worth feeding back into the next chromosome, and it
        cannot be read off the solution.

        Steps carry a fitness, not a solution, and the winner is decoded once at the end.
        A solution is a function of the partition an order cuts, so decoding the last order
        gives exactly what decoding every step along the way would have -- for one decode
        instead of one per step.
        """
        instance = solution.instance
        current = order
        fitness = solution.fitness

        # one evaluator for the whole search, local to this call so that improve stays
        # reentrant: what it holds depends on the instance alone
        evaluator = OrderEvaluator(instance)

        moved = False
        iterations = 0

        while self._max_iterations is None or iterations < self._max_iterations:
            swap = evaluator.best_swap(current, fitness, first=self._stops_at_first())
            iterations += 1

            # no neighbour beat the incumbent: this order is a local optimum for swaps
            if swap is None:
                break

            i, j, fitness = swap
            current = self._swap(current, i, j)
            moved = True

        if not moved:
            return order, iterations, solution

        return current, iterations, self._decoder.decode(instance, current)

    @abstractmethod
    def _stops_at_first(self) -> bool:
        """Whether a step takes the first improving neighbour rather than the best one."""
        pass

    def _swap(self, order: np.ndarray, i: int, j: int) -> np.ndarray:
        """The order with positions i and j swapped, as a new array."""
        swapped = order.copy()
        swapped[[i, j]] = swapped[[j, i]]

        return swapped

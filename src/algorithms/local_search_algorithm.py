from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.permutation_constructor import PermutationConstructor
from src.algorithms.permutation_decoder import PermutationDecoder
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.instance.instance import Instance

# the whole neighbourhood is O(n^2) orders and each one is decoded from scratch, so the
# work per iteration grows as n^3 -- fine up to here, hopeless on the 1000-person
# instances until the reevaluation stops redoing the untouched part of the order
MAX_PEOPLE = 100

# a swapped order and the solution it decodes to, which is what every step passes around
Step = Tuple[np.ndarray, Solution]

class LocalSearchAlgorithm(BaseAlgorithm, ABC):
    """Improves an order by swapping two people at a time, until nothing gets better.

    The neighbourhood is every swap of positions (i, j), i < j: O(n^2) orders, each scored
    by decoding it in full. Which neighbour to accept is the one thing subclasses decide,
    and they decide it by running their own loop over the neighbourhood -- best-improvement
    scans it all and keeps the best, first-improvement stops at the first gain. Splitting
    it at the loop rather than inside it keeps the O(n^2) inner loop free of any dispatch.

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

    def supports(self, instance: Instance) -> bool:
        return (
            len(instance.balances) <= MAX_PEOPLE
            and self._constructor.supports(instance)
        )

    def run(self, instance: Instance) -> RunResult:
        order, solution = self._constructor.construct(instance)
        _, improved = self.improve(order, solution)

        return RunResult(solution=improved)

    def improve(self, order: np.ndarray, solution: Solution) -> Step:
        """Refine an order until no swap improves it, or max_iterations run out.

        `solution` must be what `order` decodes to -- it is the incumbent, and its fitness
        is what every neighbour is measured against. Passing it in rather than decoding it
        here is what keeps a constructor's work from being repeated.

        The order comes back alongside the solution because a metaheuristic needs it: the
        refined order is the part worth feeding back into the next chromosome, and it
        cannot be read off the solution.
        """
        iterations = 0

        while self._max_iterations is None or iterations < self._max_iterations:
            accepted = self._accept(order, solution)

            # no neighbour beat the incumbent: this order is a local optimum for swaps
            if accepted is None:
                break

            order, solution = accepted
            iterations += 1

        return order, solution

    @abstractmethod
    def _accept(self, order: np.ndarray, solution: Solution) -> Optional[Step]:
        """The neighbour to move to, or None if none of them is strictly better."""
        pass

    def _neighbour(self, instance: Instance, order: np.ndarray, i: int, j: int) -> Step:
        """The order with positions i and j swapped, decoded.

        Decodes the whole order rather than the stretch the swap disturbs. A swap only
        perturbs the running totals between i and j, so most of the cuts are recomputed to
        the same place -- but getting the fitness right comes first, and the incremental
        version is a change to make against a correct baseline, not instead of one.
        """
        candidate = order.copy()
        candidate[i], candidate[j] = candidate[j], candidate[i]

        return candidate, self._decoder.decode(instance, candidate)

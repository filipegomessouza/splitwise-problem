from typing import List, Optional, Tuple
import numpy as np
from src.algorithms.improvement_algorithm import ImprovementAlgorithm
from src.algorithms.permutation_constructor import PermutationConstructor
from src.algorithms.solution import Solution

class VndAlgorithm(ImprovementAlgorithm):
    """Variable neighbourhood descent over a list of searches, until none improves.

    The searches are tried in order. One that improves sends the descent back to the first,
    and one that does not hands over to the next; the descent stops once every search has
    failed in a row, which leaves an order no search here improves on its own.

    Order them cheapest first: the expensive ones then run only once the cheap ones are
    spent. Each is used for its improve alone, so they need no constructor of their own --
    the VND's constructor is the one starting point they all share.
    """

    def __init__(
        self,
        constructor: Optional[PermutationConstructor],
        searches: List[ImprovementAlgorithm],
    ) -> None:
        if not searches:
            raise ValueError('VndAlgorithm needs at least one search')

        super().__init__(constructor)
        self._searches = searches

    def suffix(self) -> str:
        return 'vnd'

    def improve(self, order: np.ndarray, solution: Solution) -> Tuple[np.ndarray, int, Solution]:
        """Descend until no search improves; iterations counts the improving steps."""
        current_order = order
        current = solution
        improvements = 0
        index = 0

        while index < len(self._searches):
            improved_order, _, improved = self._searches[index].improve(current_order, current)

            if improved.fitness < current.fitness:
                current_order = improved_order
                current = improved
                improvements += 1
                index = 0
            else:
                index += 1

        return current_order, improvements, current

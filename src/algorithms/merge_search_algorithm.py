from typing import Optional, Tuple
import numpy as np
from src.algorithms.improvement_algorithm import ImprovementAlgorithm
from src.algorithms.permutation_constructor import PermutationConstructor
from src.algorithms.permutation_decoder import PermutationDecoder
from src.algorithms.solution import Solution
from src.algorithms.zero_sum_scan import first_merge, max_zero_sum_parts

class MergeSearchAlgorithm(ImprovementAlgorithm):
    """Joins two zero-sum groups and splits their union into as many zero-sum parts as it holds.

    A set of m people settles in m - k transactions at best, k being the most parts summing
    to zero it splits into, and that k is found exactly by a DP over subsets. A pair of
    groups whose union settles for less than both groups cost is replaced by those parts.
    It reaches what no swap does: turning two groups into three takes several people moving
    at once, and every swap on the way there breaks a group without forming the new ones.

    The DP is exponential in the size of the union, so a pair over max_size people is
    skipped; SplitSearchAlgorithm is the one that reaches large groups. Moves are taken
    first-improving, and the pairs rescanned after each one, until none improves.
    """

    def __init__(
        self,
        constructor: Optional[PermutationConstructor] = None,
        max_size: int = 20,
    ) -> None:
        super().__init__(constructor)
        self._max_size = max_size
        self._decoder = PermutationDecoder()

    def suffix(self) -> str:
        return 'mg'

    def improve(self, order: np.ndarray, solution: Solution) -> Tuple[np.ndarray, int, Solution]:
        """Merge until no pair improves; iterations counts the merges made."""
        instance = solution.instance
        balances = np.ascontiguousarray(instance.balances, dtype=np.int64)

        current = order
        merges = 0

        while True:
            groups = self._decoder.zero_sum_groups(balances, current)

            sizes = np.array([group.size for group in groups], dtype=np.int64)
            ends = np.cumsum(sizes)
            costs = np.array(
                [self._decoder.group_cost(balances, group) for group in groups],
                dtype=np.int64,
            )

            i, j, _ = first_merge(
                balances,
                np.ascontiguousarray(np.concatenate(groups), dtype=np.int64),
                ends - sizes,
                ends,
                costs,
                self._max_size,
            )

            if i < 0:
                break

            union = groups[i] if i == j else np.concatenate([groups[i], groups[j]])
            _, placement = max_zero_sum_parts(np.ascontiguousarray(balances[union]))

            # the parts go last, end to end, where the decoder cuts each one off on its own
            kept = [group for index, group in enumerate(groups) if index not in (i, j)]
            current = np.concatenate(kept + [union[placement]])
            merges += 1

        if merges == 0:
            return order, 0, solution

        improved = self._decoder.decode(instance, current)

        # the incumbent need not be what `order` decodes to, as with the greedy's
        if improved.fitness >= solution.fitness:
            return order, 0, solution

        return current, merges, improved

from typing import List, Optional, Tuple
import numpy as np
from src.algorithms.improvement_algorithm import ImprovementAlgorithm
from src.algorithms.permutation_constructor import PermutationConstructor
from src.algorithms.permutation_decoder import PermutationDecoder
from src.algorithms.solution import Solution
from src.algorithms.zero_sum_scan import subset_sums

class SplitSearchAlgorithm(ImprovementAlgorithm):
    """Splits zero-sum groups along the smaller zero-sum subsets they hold.

    A group of m people costs m - 1 transactions unless some of them sum to zero on their
    own, and then splitting them off saves one. The greedy that settles a group finds such
    a subset only when two of its running amounts happen to match, so large groups often
    keep many it never sees.

    Each group is split as long as one of its parts holds a zero-sum subset, found by a
    subset-sum DP: the sums its positives reach and the sums its negatives reach, and the
    smallest both reach. That costs the size of the group times its total rather than
    2^size, which is what lets this reach the large groups an exact repartition cannot.
    It finds a split, not the best one, so a group is replaced by its parts only when they
    settle for less than the greedy settles the whole group in.
    """

    def __init__(self, constructor: Optional[PermutationConstructor] = None) -> None:
        super().__init__(constructor)
        self._decoder = PermutationDecoder()

    def suffix(self) -> str:
        return 'sp'

    def improve(self, order: np.ndarray, solution: Solution) -> Tuple[np.ndarray, int, Solution]:
        """Split every group that settles for less in parts; iterations counts the groups split."""
        instance = solution.instance
        balances = instance.balances

        groups: List[np.ndarray] = []
        splits = 0

        for group in self._decoder.zero_sum_groups(balances, order):
            parts = self._split(group, balances)
            cost = sum(self._decoder.group_cost(balances, part) for part in parts)

            if len(parts) > 1 and cost < self._decoder.group_cost(balances, group):
                groups.extend(parts)
                splits += 1
            else:
                groups.append(group)

        if splits == 0:
            return order, 0, solution

        # each part sums to zero, so laid end to end they are cut exactly where they meet
        improved_order = np.concatenate(groups)
        improved = self._decoder.decode(instance, improved_order)

        # the incumbent need not be what `order` decodes to, as with the greedy's
        if improved.fitness >= solution.fitness:
            return order, 0, solution

        return improved_order, splits, improved

    def _split(self, group: np.ndarray, balances: np.ndarray) -> List[np.ndarray]:
        """`group` split into parts none of which holds a zero-sum subset any more."""
        done: List[np.ndarray] = []
        pending = [group]

        while pending:
            part = pending.pop()
            subset = self._zero_sum_subset(balances[part])

            if subset is None:
                done.append(part)
            else:
                pending.extend([part[subset], part[~subset]])

        return done

    def _zero_sum_subset(self, balances: np.ndarray) -> Optional[np.ndarray]:
        """A mask over a zero-sum set picking a proper, non-empty subset that sums to zero.

        Picks the smallest sum both sides reach. A subset reaching s leaves a complement
        reaching the total minus s, so sums past half the total add nothing.
        """
        positive = np.flatnonzero(balances > 0)
        negative = np.flatnonzero(balances < 0)

        credits = np.ascontiguousarray(balances[positive], dtype=np.int64)
        debts = np.ascontiguousarray(-balances[negative], dtype=np.int64)
        limit = int(credits.sum()) // 2

        reached_credits, first_credit = subset_sums(credits, limit)
        reached_debts, first_debt = subset_sums(debts, limit)

        # sum 0 is the empty subset, which every side reaches and no split comes from
        common = np.flatnonzero(reached_credits[1:] & reached_debts[1:])

        if common.size == 0:
            return None

        total = int(common[0]) + 1

        subset = np.zeros(balances.size, dtype=bool)
        subset[positive[self._pick(credits, first_credit, total)]] = True
        subset[negative[self._pick(debts, first_debt, total)]] = True

        return subset

    def _pick(self, values: np.ndarray, first: np.ndarray, total: int) -> List[int]:
        """Indices of values adding up to `total`, read back from subset_sums' `first`."""
        picked: List[int] = []

        while total > 0:
            index = int(first[total])
            picked.append(index)
            total -= int(values[index])

        return picked

from typing import List, Tuple
import heapq
import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.instance.instance import Instance

EMPTY = np.empty(0, dtype=np.int64)

class GreedyAlgorithm(BaseAlgorithm):
    def name(self) -> str:
        return 'greedy'

    def run(self, instance: Instance) -> RunResult:
        payers, receivers, amounts, remaining = self.get_balances_without_direct_transactions(instance)

        heap_payers, heap_receivers, heap_amounts = self._settle_largest_first(instance, remaining)

        solution = Solution(
            instance=instance,
            payers=np.concatenate((payers, heap_payers)),
            receivers=np.concatenate((receivers, heap_receivers)),
            amounts=np.concatenate((amounts, heap_amounts)),
        )

        return RunResult(solution=solution, status='heuristic')

    def get_balances_without_direct_transactions(
        self, instance: Instance
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pair people who owe and are owed the very same amount, one transaction each.

        Sorting each side by amount puts equal amounts in a contiguous run, so a person
        pairs off exactly when the other side holds at least as many people at that
        amount as the person's own position within its run. Which specific pair of people
        gets matched differs from a bucket-and-pop pass, but the count -- the only part
        the fitness sees -- is the same.

        Returns the matched transactions plus the people still holding a balance.
        """
        balances = instance.balances

        receivers = np.flatnonzero(balances > 0)
        payers = np.flatnonzero(balances < 0)

        receiver_amounts = balances[receivers]
        payer_amounts = -balances[payers]

        receivers, receiver_amounts = self._sort_by_amount(receivers, receiver_amounts)
        payers, payer_amounts = self._sort_by_amount(payers, payer_amounts)

        matched_receivers = self._rank_within_amount(receiver_amounts) < self._count_at(
            payer_amounts, receiver_amounts
        )
        matched_payers = self._rank_within_amount(payer_amounts) < self._count_at(
            receiver_amounts, payer_amounts
        )

        # both sides are ordered by amount and then by position within it, and each amount
        # contributes min(receivers, payers) to either side, so they line up element-wise
        remaining = np.concatenate((receivers[~matched_receivers], payers[~matched_payers]))

        return (
            payers[matched_payers],
            receivers[matched_receivers],
            receiver_amounts[matched_receivers],
            remaining,
        )

    def _settle_largest_first(
        self, instance: Instance, remaining: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Repeatedly settle the largest payer against the largest receiver.

        Deliberately not vectorised, and deliberately on Python ints: every step depends
        on the previous one, so there is no array-wide equivalent, and np.int64 scalars
        measured 23% slower than Python ints through heapq -- on the greedy's dominant
        cost. tolist() is the boundary where the arrays are handed back.
        """
        balances = instance.balances[remaining].tolist()
        people = remaining.tolist()

        # heapq is a min-heap, so magnitudes are stored negated to pop the largest first;
        # the person index rides along and breaks ties deterministically
        payers = [(balance, person) for person, balance in zip(people, balances) if balance < 0]
        receivers = [(-balance, person) for person, balance in zip(people, balances) if balance > 0]

        heapq.heapify(payers)
        heapq.heapify(receivers)

        settled_payers: List[int] = []
        settled_receivers: List[int] = []
        settled_amounts: List[int] = []

        while payers and receivers:
            owed, payer = heapq.heappop(payers)
            due, receiver = heapq.heappop(receivers)

            owed = -owed
            due = -due
            amount = min(owed, due)

            settled_payers.append(payer)
            settled_receivers.append(receiver)
            settled_amounts.append(amount)

            if owed > due:
                heapq.heappush(payers, (-(owed - due), payer))
            elif owed < due:
                heapq.heappush(receivers, (-(due - owed), receiver))

        # three flat lists convert far faster than one list of triples would
        return (
            np.array(settled_payers, dtype=np.int64),
            np.array(settled_receivers, dtype=np.int64),
            np.array(settled_amounts, dtype=np.int64),
        )

    @staticmethod
    def _sort_by_amount(people: np.ndarray, amounts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        order = np.argsort(amounts, kind='stable')

        return people[order], amounts[order]

    @staticmethod
    def _rank_within_amount(amounts: np.ndarray) -> np.ndarray:
        """Position of each person among the people sharing its amount, counting from 0."""
        if amounts.size == 0:
            return EMPTY

        starts = np.flatnonzero(np.concatenate(([True], amounts[1:] != amounts[:-1])))
        counts = np.diff(np.append(starts, amounts.size))

        return np.arange(amounts.size) - np.repeat(starts, counts)

    @staticmethod
    def _count_at(sorted_amounts: np.ndarray, wanted: np.ndarray) -> np.ndarray:
        """How many entries of sorted_amounts equal each entry of wanted."""
        return (
            np.searchsorted(sorted_amounts, wanted, side='right')
            - np.searchsorted(sorted_amounts, wanted, side='left')
        )

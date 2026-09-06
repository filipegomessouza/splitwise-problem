from typing import List, Tuple
import heapq
import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.helpers.direct_transactions import pair_direct_transactions
from src.instance.instance import Instance

Transactions = Tuple[np.ndarray, np.ndarray, np.ndarray]

class GreedyAlgorithm(BaseAlgorithm):
    def name(self) -> str:
        return 'greedy'

    def run(self, instance: Instance) -> RunResult:
        payers, receivers, amounts = self.settle(
            np.arange(len(instance.balances)), instance.balances
        )

        solution = Solution(
            instance=instance,
            payers=payers,
            receivers=receivers,
            amounts=amounts,
        )

        return RunResult(solution=solution)

    def settle(self, people: np.ndarray, balances: np.ndarray) -> Transactions:
        """Settle any set of people whose balances sum to zero.

        Works on a subset as readily as on a whole instance, so a decoder that has split
        an instance into independent zero-sum groups can hand each one over without
        building an Instance around it. Internally everything is a position into
        `balances`; `people` carries those positions back to the caller's index space,
        applied once at the end.

        Two phases: settle whoever can be paired off exactly, then settle the rest largest
        against largest. A caller that has already paired can skip straight to the second.
        """
        paired_payers, paired_receivers, paired_amounts, left = pair_direct_transactions(balances)

        heap_payers, heap_receivers, heap_amounts = self.settle_largest_first(
            left, balances[left]
        )

        return (
            people[np.concatenate((paired_payers, heap_payers))],
            people[np.concatenate((paired_receivers, heap_receivers))],
            np.concatenate((paired_amounts, heap_amounts)),
        )

    def settle_largest_first(self, positions: np.ndarray, balances: np.ndarray) -> Transactions:
        """Repeatedly settle the largest payer against the largest receiver.

        Deliberately not vectorised, and deliberately on Python ints: every step depends
        on the previous one, so there is no array-wide equivalent, and np.int64 scalars
        measured 23% slower than Python ints through heapq -- on the greedy's dominant
        cost. tolist() is the boundary where the arrays are handed back.
        """
        owed = balances.tolist()
        people = positions.tolist()

        # heapq is a min-heap, so magnitudes are stored negated to pop the largest first;
        # the position rides along and breaks ties deterministically
        payers = [(balance, person) for person, balance in zip(people, owed) if balance < 0]
        receivers = [(-balance, person) for person, balance in zip(people, owed) if balance > 0]

        heapq.heapify(payers)
        heapq.heapify(receivers)

        settled_payers: List[int] = []
        settled_receivers: List[int] = []
        settled_amounts: List[int] = []

        while payers and receivers:
            owes, payer = heapq.heappop(payers)
            due, receiver = heapq.heappop(receivers)

            owes = -owes
            due = -due
            amount = min(owes, due)

            settled_payers.append(payer)
            settled_receivers.append(receiver)
            settled_amounts.append(amount)

            if owes > due:
                heapq.heappush(payers, (-(owes - due), payer))
            elif owes < due:
                heapq.heappush(receivers, (-(due - owes), receiver))

        # three flat lists convert far faster than one list of triples would
        return (
            np.array(settled_payers, dtype=np.int64),
            np.array(settled_receivers, dtype=np.int64),
            np.array(settled_amounts, dtype=np.int64),
        )

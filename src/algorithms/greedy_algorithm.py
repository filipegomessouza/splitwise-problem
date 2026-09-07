from typing import List, Optional, Tuple
import heapq
import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.random_keys import RandomKeys
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.helpers.direct_transactions import pair_direct_transactions
from src.instance.instance import Instance

Transactions = Tuple[np.ndarray, np.ndarray, np.ndarray]

class GreedyAlgorithm(BaseAlgorithm):
    """Settles everyone in one pass, largest debt against largest credit.

    Also serves as a starting point for a local search, which is what the keys are for.
    They take no part in the solution -- one heap over every survivor is the whole of the
    algorithm -- and exist only so that construct can hand out an order.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._keys = RandomKeys(seed)

    def name(self) -> str:
        return 'greedy'

    @property
    def random_keys(self) -> Optional[np.ndarray]:
        return self._keys.keys

    def set_random_keys(self, random_keys: Optional[np.ndarray]) -> None:
        """Fix the order construct hands out, without touching what run produces.

        The same surface as RandomKeyAlgorithm on purpose: giving both constructors the
        very same chromosome is what makes the permutation a controlled variable when the
        two are compared as starting points.
        """
        self._keys.set(random_keys)

    def supports(self, instance: Instance) -> bool:
        return self._keys.fits(len(instance.balances))

    def run(self, instance: Instance) -> RunResult:
        return RunResult(solution=self._settle(instance))

    def construct(self, instance: Instance) -> Tuple[np.ndarray, Solution]:
        """An order over the survivors, and this algorithm's own solution.

        The pair is deliberately inconsistent: the solution is not what the order decodes
        to, and no order would decode to it. A decoder cuts wherever a running total
        repeats, while this settles every survivor as a single group -- reproducing that
        would take an order in which no proper contiguous block sums to zero, which is a
        search problem of its own and may have no answer.

        So the order says where a local search should start looking, and the solution says
        what it has to beat. LocalSearchAlgorithm.improve documents what that costs.
        """
        balances = instance.balances

        # pair_direct_transactions rather than the decoder's survivors: PermutationDecoder
        # is built on this class, and reaching back for it would close an import cycle
        _, _, _, left = pair_direct_transactions(balances)

        return self._keys.order(len(balances), left), self._settle(instance)

    def _settle(self, instance: Instance) -> Solution:
        payers, receivers, amounts = self.settle(
            np.arange(len(instance.balances)), instance.balances
        )

        return Solution(
            instance=instance,
            payers=payers,
            receivers=receivers,
            amounts=amounts,
        )

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

    def count_largest_first(self, balances: List[int]) -> int:
        """How many transactions settle_largest_first would need, without building them.

        Takes balances alone, no people: what decides the count is where a pop finds owes
        equal to due, since that step settles two people rather than one, and that depends
        on the magnitudes only. Who is who changes which pair settles, never how many do.

        For a search that scores thousands of neighbours a second, the three lists and the
        three arrays settle_largest_first ends with are the bulk of its cost, and every one
        of them is thrown away unread.
        """
        payers = [balance for balance in balances if balance < 0]
        receivers = [-balance for balance in balances if balance > 0]

        heapq.heapify(payers)
        heapq.heapify(receivers)

        count = 0

        while payers and receivers:
            owes = -heapq.heappop(payers)
            due = -heapq.heappop(receivers)

            count += 1

            if owes > due:
                heapq.heappush(payers, -(owes - due))
            elif owes < due:
                heapq.heappush(receivers, -(due - owes))

        return count

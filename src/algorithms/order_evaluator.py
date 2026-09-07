from typing import Dict, FrozenSet, List, Tuple
from src.algorithms.greedy_algorithm import GreedyAlgorithm
from src.algorithms.zero_sum_scan import State, ZeroSumScan
from src.helpers.direct_transactions import pair_direct_transactions
from src.instance.instance import Instance

# the scan's state at some position, and the cost of everything it closed before reaching it
Prefix = Tuple[State, int]

class OrderEvaluator:
    """Scores orders of one instance's survivors, as a transaction count and nothing else.

    A local search reads a single integer per neighbour, so building a Solution for each one
    -- pairing the instance again, cutting groups into numpy arrays, settling them into three
    arrays and concatenating those -- is almost all waste. This keeps only what the integer
    needs.

    Three things make a swap far cheaper to score than to decode:

    - a group's cost depends on its members as a set, not on their order, because the greedy
      heapifies them. So the cost of any group ever seen is worth caching, and a swap that
      leaves a group's membership alone gets its cost for free;
    - a swap at positions i and j shifts the running totals only within (i, j], and by a
      constant, so repeats inside that window survive it and only those crossing i or j
      break. The decomposition moves at the edges, which is why the cache hits;
    - the scan up to i is the same order for every j, so it is worth scanning once per i and
      restoring, rather than rescanning per neighbour.

    Worth keeping for a whole search rather than a single neighbourhood: measured over a
    100-person instance, 96.5% of cost lookups hit, the cache peaked at 712 entries, and
    building the frozenset key measured 22x cheaper than the heap pass it stands in for.
    """

    def __init__(self, instance: Instance) -> None:
        balances = instance.balances

        # once, not once per neighbour: the pairing depends on the balances alone, and no
        # order can change it
        paired_payers, _, _, _ = pair_direct_transactions(balances)

        self._pairs = len(paired_payers)
        self._balances = balances.tolist()
        self._greedy = GreedyAlgorithm()
        self._scan = ZeroSumScan(balances)
        self._costs: Dict[FrozenSet[int], int] = {}

    @property
    def balances(self) -> List[int]:
        return self._balances

    def fitness(self, order: List[int]) -> int:
        """The transaction count of an order, scanned from the start."""
        self._scan.reset()

        return self._pairs + self._closed(order)

    def prefixes(self, order: List[int]) -> List[Prefix]:
        """A restorable prefix for every position a swap can start at.

        One pass builds them all, so the whole neighbourhood's prefix work is O(n) instead
        of the O(n^2) it would cost to advance to each i separately. The last position is
        left out: a swap needs a partner after it.
        """
        self._scan.reset()

        cost = 0
        prefixes: List[Prefix] = [(self._scan.snapshot(), cost)]

        for person in order[:-2]:
            group = self._scan.add(person)

            if group is not None:
                cost += self._cost(group)

            prefixes.append((self._scan.snapshot(), cost))

        return prefixes

    def fitness_from(self, prefix: Prefix, order: List[int], i: int, j: int) -> int:
        """The transaction count of `order` with positions i and j swapped.

        `prefix` must be prefixes(order)[i]. Only the tail is scanned, and the swap is
        applied to a copy of it rather than to the order, so the caller's order survives the
        whole neighbourhood untouched.
        """
        state, cost = prefix

        self._scan.restore(state)

        tail = order[i:]
        tail[0] = order[j]
        tail[j - i] = order[i]

        return self._pairs + cost + self._closed(tail)

    def _closed(self, people: List[int]) -> int:
        """Feed people to the scan and total up what the groups it closes cost."""
        scan = self._scan
        cost = 0

        for person in people:
            group = scan.add(person)

            if group is not None:
                cost += self._cost(group)

        return cost

    def _cost(self, group: List[int]) -> int:
        """How many transactions a group costs, from the cache where possible.

        Keyed by the members as a frozenset, which is exactly what the cost depends on. An
        integer hash accumulated as the scan runs would be cheaper still, but it would make
        the fitness answer to a hash collision, and building the set is not what costs here.
        """
        key = frozenset(group)
        cost = self._costs.get(key)

        if cost is None:
            balances = self._balances
            cost = self._greedy.count_largest_first([balances[person] for person in group])
            self._costs[key] = cost

        return cost

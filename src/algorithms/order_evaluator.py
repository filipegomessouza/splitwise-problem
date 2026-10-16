from typing import Optional, Tuple
import numpy as np
from src.algorithms.zero_sum_scan import best_swap
from src.helpers.direct_transactions import pair_direct_transactions
from src.instance.instance import Instance

# an improving swap (i, j) and what the order scores with it
Swap = Tuple[int, int, int]

class OrderEvaluator:
    """Scores the swap neighbourhood of an order, as a transaction count and nothing else.

    A local search reads a single integer per neighbour, so building a Solution for each one
    -- pairing the instance again, cutting groups into numpy arrays, settling them into three
    arrays and concatenating those -- is almost all waste. This keeps only what the integer
    needs, and runs the whole neighbourhood as one compiled call.

    Three things make a swap far cheaper to score than to decode:

    - a group's cost depends on its members as a set, not on their order, because the greedy
      heapifies them. So the cost of any group ever seen is worth caching, and a swap that
      leaves a group's membership alone gets its cost for free;
    - a swap at positions i and j shifts the running totals only within (i, j], and by a
      constant, so repeats inside that window survive it and only those crossing i or j
      break. The decomposition moves at the edges, which is why the cache hits;
    - the scan up to i is the same order for every j, so it is worth scanning once per i and
      building on, rather than rescanning per neighbour.

    The mechanics live in zero_sum_scan.best_swap; what this holds is what never changes
    over a search, worked out once.
    """

    def __init__(self, instance: Instance) -> None:
        # contiguous, because the compiled scan is declared over contiguous arrays, and a
        # reader may hand over a column sliced out of a wider file
        self._balances = np.ascontiguousarray(instance.balances, dtype=np.int64)

        # once, not once per neighbour: the pairing depends on the balances alone, and no
        # order can change it
        paired_payers, _, _, _ = pair_direct_transactions(self._balances)
        self._pairs = len(paired_payers)

        # one random key per person, xor-ed into a hash of each group's members. Fixed, since
        # they only spread the cache and a hit is confirmed member by member regardless
        self._keys = np.random.default_rng(0).integers(
            np.iinfo(np.int64).min,
            np.iinfo(np.int64).max,
            size=self._balances.size,
            dtype=np.int64,
            endpoint=True,
        )

    def best_swap(self, order: np.ndarray, fitness: int, first: bool) -> Optional[Swap]:
        """The swap that beats `fitness` by the most -- or with `first`, the first that does.

        None if no swap is strictly better, which makes `order` a local optimum for swaps.
        """
        i, j, swapped_fitness = best_swap(
            self._balances,
            np.ascontiguousarray(order, dtype=np.int64),
            fitness,
            self._pairs,
            self._keys,
            first,
        )

        if i < 0:
            return None

        return i, j, swapped_fitness

from typing import Dict, List, Tuple
import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.greedy_algorithm import GreedyAlgorithm
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.instance.instance import Instance

class RandomKeyAlgorithm(BaseAlgorithm):
    """Builds a solution from a vector of random keys, one per person.

    The keys order the people; that order is split into groups whose balances sum to
    zero, and each group is an independent settlement solved by the greedy. Splitting
    pays off because a group of m people costs at most m - 1 transactions, so k groups
    cap the whole solution at N - k.

    Written as a BRKGA decoder, so the object is reusable: set_random_keys swaps the
    chromosome without rebuilding anything, and one GreedyAlgorithm is shared across
    every group of every call.
    """

    def __init__(self, random_keys: np.ndarray) -> None:
        self._greedy = GreedyAlgorithm()
        self.set_random_keys(random_keys)

    def name(self) -> str:
        return 'random_key'

    def set_random_keys(self, random_keys: np.ndarray) -> None:
        keys = np.asarray(random_keys, dtype=np.float64)

        if keys.ndim != 1:
            raise ValueError('random_keys must be one-dimensional')

        if keys.size and (keys.min() < 0.0 or keys.max() > 1.0):
            raise ValueError('random_keys must lie between 0 and 1')

        # no length check on purpose: swapping in a chromosome of another length is how
        # one object serves instances of different sizes
        self._random_keys = keys

    def supports(self, instance: Instance) -> bool:
        return len(instance.balances) == len(self._random_keys)

    def run(self, instance: Instance) -> RunResult:
        if not self.supports(instance):
            raise ValueError(
                f"{len(self._random_keys)} random keys cannot decode an instance of "
                f"{len(instance.balances)} people"
            )

        balances = instance.balances

        # stable so that equal keys keep person order, which keeps a run reproducible
        order = np.argsort(self._random_keys, kind='stable')

        payers: List[np.ndarray] = []
        receivers: List[np.ndarray] = []
        amounts: List[np.ndarray] = []

        for group in self.zero_sum_groups(balances, order):
            group_payers, group_receivers, group_amounts = self._greedy.settle(
                group, balances[group]
            )

            payers.append(group_payers)
            receivers.append(group_receivers)
            amounts.append(group_amounts)

        solution = Solution(
            instance=instance,
            payers=np.concatenate(payers) if payers else np.empty(0, dtype=np.int64),
            receivers=np.concatenate(receivers) if receivers else np.empty(0, dtype=np.int64),
            amounts=np.concatenate(amounts) if amounts else np.empty(0, dtype=np.int64),
        )

        return RunResult(solution=solution, status='constructive')

    def zero_sum_groups(self, balances: np.ndarray, order: np.ndarray) -> List[np.ndarray]:
        """Split the people, in the given order, into groups whose balances sum to zero.

        The balances between two positions sum to zero exactly when the running total is
        the same at both -- the value they share is irrelevant, only that it repeats. So
        every repeat of a running total closes a group. Cutting only where the total is
        zero, which is the special case of repeating the empty prefix, would throw away
        every other repeat.

        Removing a group changes nothing about the totals that follow it, since the group
        contributed zero, so a single pass suffices: no need to restart on the remainder.
        """
        pending: List[int] = []
        # running total -> how far into `pending` it was reached
        opened_at: Dict[int, int] = {0: 0}
        # insertion order, so a closed group's entries can be rolled back
        history: List[Tuple[int, int]] = [(0, 0)]

        running = 0
        groups: List[np.ndarray] = []

        for person in order:
            running += int(balances[person])
            pending.append(int(person))

            if running in opened_at:
                at = opened_at[running]

                groups.append(np.array(pending[at:], dtype=np.int64))
                del pending[at:]

                # the positions recorded inside the group are gone; leaving them behind
                # would let a later repeat cut at an index that now holds someone else
                while history[-1][1] > at:
                    stale, _ = history.pop()
                    del opened_at[stale]
            else:
                opened_at[running] = len(pending)
                history.append((running, len(pending)))

        return groups

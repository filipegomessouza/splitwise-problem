from typing import Dict, List, Optional, Tuple
import numpy as np

# pending, opened_at, history and the running total, which is the whole of the scan's state
State = Tuple[List[int], Dict[int, int], List[Tuple[int, int]], int]

class ZeroSumScan:
    """Splits people, in the order they are added, into groups whose balances sum to zero.

    The balances between two positions sum to zero exactly when the running total is the
    same at both -- the value they share is irrelevant, only that it repeats. So every
    repeat of a running total closes a group. Cutting only where the total is zero, which is
    the special case of repeating the empty prefix, would throw away every other repeat.

    Removing a group changes nothing about the totals that follow it, since the group
    contributed zero, so a single pass suffices: no need to restart on the remainder.

    Written as an object rather than a function over a whole order because a neighbourhood
    of swaps needs it paused: every swap at position i leaves the scan up to i untouched, so
    that prefix is worth scanning once and restoring, not rescanning once per neighbour.
    """

    def __init__(self, balances: np.ndarray) -> None:
        # a Python list, because add is called once per person per neighbour and pulling a
        # scalar out of a numpy array costs several times what indexing a list does
        self._balances = balances.tolist()

        self.reset()

    def reset(self) -> None:
        self._pending: List[int] = []
        # running total -> how far into `pending` it was reached
        self._opened_at: Dict[int, int] = {0: 0}
        # insertion order, so a closed group's entries can be rolled back
        self._history: List[Tuple[int, int]] = [(0, 0)]
        self._running = 0

    def add(self, person: int) -> Optional[List[int]]:
        """Extend the order by one person, returning the group they closed, if any."""
        self._running += self._balances[person]
        self._pending.append(person)

        if self._running in self._opened_at:
            at = self._opened_at[self._running]

            group = self._pending[at:]
            del self._pending[at:]

            # the positions recorded inside the group are gone; leaving them behind would
            # let a later repeat cut at an index that now holds someone else
            while self._history[-1][1] > at:
                stale, _ = self._history.pop()
                del self._opened_at[stale]

            return group

        self._opened_at[self._running] = len(self._pending)
        self._history.append((self._running, len(self._pending)))

        return None

    def snapshot(self) -> State:
        """The state as it stands, detached from further calls to add."""
        return (
            self._pending.copy(),
            self._opened_at.copy(),
            self._history.copy(),
            self._running,
        )

    def restore(self, state: State) -> None:
        """Rewind to a snapshot, which stays good for restoring again.

        Copies rather than adopts: one snapshot is restored once per neighbour, so letting
        add mutate it would corrupt every neighbour after the first.
        """
        pending, opened_at, history, running = state

        self._pending = pending.copy()
        self._opened_at = opened_at.copy()
        self._history = history.copy()
        self._running = running

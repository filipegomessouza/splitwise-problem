from typing import List
import numpy as np
from src.algorithms.greedy_algorithm import GreedyAlgorithm
from src.algorithms.solution import Solution
from src.algorithms.zero_sum_scan import ZeroSumScan
from src.helpers.direct_transactions import pair_direct_transactions
from src.instance.instance import Instance

class PermutationDecoder:
    """Turns an order of people into the settlement that order implies.

    The order is split into groups whose balances sum to zero, and each group is an
    independent settlement solved by the greedy. Splitting pays off because a group of m
    people costs at most m - 1 transactions, so k groups cap the whole solution at N - k
    -- which makes the number of cuts, not the order itself, what a search over orders is
    really after.

    Shared by everything that scores a permutation, so that a constructor and a local
    search cannot drift into two different notions of fitness for the same order. Holds no
    per-instance state: one decoder serves a whole benchmark, and one GreedyAlgorithm is
    reused across every group of every call.
    """

    def __init__(self) -> None:
        self._greedy = GreedyAlgorithm()

    def survivors(self, balances: np.ndarray) -> np.ndarray:
        """The people left holding a balance once every exact match is paired off.

        These, and only these, are what an order has to cover: the paired ones are settled
        the same way whatever the order says, so ordering them decides nothing.
        """
        _, _, _, left = pair_direct_transactions(balances)

        return left

    def decode(self, instance: Instance, order: np.ndarray) -> Solution:
        """Settle the instance by cutting `order` into zero-sum groups.

        `order` must be a permutation of survivors(instance.balances), in instance indices.
        Nothing checks it: the caller that built the order knows, and a decoder called once
        per neighbour of an O(n^2) neighbourhood is no place to re-derive it.

        The pairing is repeated on every call even though it never changes, which keeps the
        decoder stateless and reentrant. It is fully vectorised, while the cuts and the
        greedy are sequential Python -- so it is not what this costs.
        """
        balances = instance.balances

        # paired before splitting, not within each group: a +v and its -v can land in
        # different groups, and then neither gets its free direct transaction. Every pair
        # is a zero-sum set of two, so this hands out parts and shrinks what is left to
        # partition
        paired_payers, paired_receivers, paired_amounts, _ = pair_direct_transactions(balances)

        payers: List[np.ndarray] = [paired_payers]
        receivers: List[np.ndarray] = [paired_receivers]
        amounts: List[np.ndarray] = [paired_amounts]

        for group in self.zero_sum_groups(balances, order):
            # settle_largest_first rather than settle: pairing above exhausted every exact
            # match, so each magnitude now has people on one side only and a second
            # pairing pass over a group would provably find nothing
            group_payers, group_receivers, group_amounts = self._greedy.settle_largest_first(
                group, balances[group]
            )

            payers.append(group_payers)
            receivers.append(group_receivers)
            amounts.append(group_amounts)

        # no empty-list guard needed: the pairing above always contributes a first entry,
        # even when it is an empty array
        return Solution(
            instance=instance,
            payers=np.concatenate(payers),
            receivers=np.concatenate(receivers),
            amounts=np.concatenate(amounts),
        )

    def zero_sum_groups(self, balances: np.ndarray, order: np.ndarray) -> List[np.ndarray]:
        """Split the people, in the given order, into groups whose balances sum to zero.

        Runs a ZeroSumScan from end to end and collects what it closes; the cuts themselves
        are defined there, once, so that a search scoring neighbours and this cannot drift
        into disagreeing about where a group ends.
        """
        scan = ZeroSumScan(balances)
        groups: List[np.ndarray] = []

        for person in order.tolist():
            group = scan.add(person)

            if group is not None:
                groups.append(np.array(group, dtype=np.int64))

        return groups

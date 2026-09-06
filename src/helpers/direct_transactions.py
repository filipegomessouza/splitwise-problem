from typing import Tuple
import numpy as np

EMPTY = np.empty(0, dtype=np.int64)

Pairing = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def pair_direct_transactions(balances: np.ndarray) -> Pairing:
    """Pair people who owe and are owed the very same amount, one transaction each.

    Sorting each side by amount puts equal amounts in a contiguous run, so a person pairs
    off exactly when the other side holds at least as many people at that amount as the
    person's own position within its run. Which specific pair of people gets matched
    differs from a bucket-and-pop pass, but the count -- the only part the fitness sees --
    is the same.

    Returns the matched transactions plus the people still holding a balance, all as
    positions into `balances`.
    """
    receivers = np.flatnonzero(balances > 0)
    payers = np.flatnonzero(balances < 0)

    receiver_amounts = balances[receivers]
    payer_amounts = -balances[payers]

    receivers, receiver_amounts = _sort_by_amount(receivers, receiver_amounts)
    payers, payer_amounts = _sort_by_amount(payers, payer_amounts)

    matched_receivers = _rank_within_amount(receiver_amounts) < _count_at(
        payer_amounts, receiver_amounts
    )
    matched_payers = _rank_within_amount(payer_amounts) < _count_at(
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

def _sort_by_amount(people: np.ndarray, amounts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(amounts, kind='stable')

    return people[order], amounts[order]

def _rank_within_amount(amounts: np.ndarray) -> np.ndarray:
    """Position of each person among the people sharing its amount, counting from 0."""
    if amounts.size == 0:
        return EMPTY

    starts = np.flatnonzero(np.concatenate(([True], amounts[1:] != amounts[:-1])))
    counts = np.diff(np.append(starts, amounts.size))

    return np.arange(amounts.size) - np.repeat(starts, counts)

def _count_at(sorted_amounts: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """How many entries of sorted_amounts equal each entry of wanted."""
    return (
        np.searchsorted(sorted_amounts, wanted, side='right')
        - np.searchsorted(sorted_amounts, wanted, side='left')
    )

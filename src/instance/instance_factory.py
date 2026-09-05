from src.instance.instance import Instance
from typing import List, Optional
import os
import random

# a zero-sum block needs one payer and one receiver at the very least
MIN_BLOCK_SIZE = 2

class InstanceFactory:
    """Draws balance vectors with a planted partition into zero-sum blocks.

    The optimum of the problem is people - k*, where k* is the largest number of parts
    in a partition of the balances into zero-sum subsets. Drawing balances at random
    leaves k* to chance, so instead the vector is assembled block by block: `structure`
    picks how many zero-sum blocks to plant, which makes k* at least that many and puts
    a known ceiling on the optimum. Accidental zero-sum subsets can only push k* higher,
    so the planted count is a lower bound, not an exact figure -- and their number grows
    like 2^people / (2 * people * max_balance), so keeping the low-structure regime
    meaningful takes a max_balance that scales with people.
    """

    def __init__(
        self,
        people: int,
        max_balance: int,
        structure: float,
        seed: Optional[int] = None,
    ) -> None:
        if people < 2:
            raise ValueError('people must be at least 2')

        if max_balance < 1:
            raise ValueError('max_balance must be at least 1')

        if not 0.0 <= structure <= 1.0:
            raise ValueError('structure must be between 0 and 1')

        self._people = people
        self._max_balance = max_balance
        self._structure = structure
        self._rng = random.Random(seed)

    def create(self) -> Instance:
        balances: List[int] = []

        for size in self._block_sizes():
            balances.extend(self._zero_sum_block(size))

        self._rng.shuffle(balances)

        return Instance(balances)

    def create_as_txt(self, file_path: str) -> None:
        instance = self.create()

        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, 'w') as file:
            for balance in instance.balances:
                file.write(f"{balance}\n")

    def _block_sizes(self) -> List[int]:
        """Split the people into blocks, from one big block at structure 0 to all pairs at 1."""
        most_blocks = self._people // MIN_BLOCK_SIZE
        blocks = 1 + round(self._structure * (most_blocks - 1))

        sizes = [MIN_BLOCK_SIZE] * blocks

        for _ in range(self._people - blocks * MIN_BLOCK_SIZE):
            sizes[self._rng.randrange(blocks)] += 1

        return sizes

    def _zero_sum_block(self, size: int) -> List[int]:
        """Draw `size` non-zero balances within the range that sum to exactly zero."""
        receivers = self._receiver_count(size)
        payers = size - receivers

        # the block's volume: bounded so that both sides can represent it, since each
        # side splits it into parts of at least 1 and at most max_balance
        volume = self._rng.randint(
            max(receivers, payers),
            min(receivers, payers) * self._max_balance,
        )

        return (
            self._bounded_split(volume, receivers)
            + [-part for part in self._bounded_split(volume, payers)]
        )

    def _receiver_count(self, size: int) -> int:
        """Pick how many of a block's people are receivers, by coin flip within reach.

        Both sides carry the same volume, so a side of k people covers between k and
        k * max_balance -- which leaves the split feasible only where the larger side
        fits inside what the smaller side can reach. That window is the whole of
        [1, size - 1] unless max_balance is small enough to rival the block size.
        """
        span = self._max_balance + 1

        # from size - receivers <= receivers * max_balance and its mirror image
        fewest = max(1, -(-size // span))
        most = min(size - 1, size * self._max_balance // span)

        if fewest > most:
            raise ValueError(
                f"max_balance {self._max_balance} is too small to split a block of "
                f"{size} people into two sides that can settle each other"
            )

        return min(max(self._coin_flips(size), fewest), most)

    def _bounded_split(self, total: int, parts: int) -> List[int]:
        """Split `total` into `parts` integers in [1, max_balance], one at a time.

        Each draw is bounded by what still leaves the remaining parts representable, so
        the split always succeeds as long as parts <= total <= parts * max_balance.

        Drawing uniformly within those bounds would spend the budget at the wrong rate
        whenever the average part is far from max_balance / 2, and the bounds would then
        pin the whole tail of the split to a single value -- which shows up as a pile of
        balances all equal to 1 or all equal to max_balance, exactly the repeats a greedy
        pairing settles for free. So each draw is shaped to have the average still to be
        spent as its mean, which keeps the budget on track and lets no such tail form.
        """
        split: List[int] = []

        for index in range(parts):
            remaining = parts - 1 - index

            low = max(1, total - remaining * self._max_balance)
            high = min(self._max_balance, total - remaining)

            part = self._draw_with_mean(low, high, total / (remaining + 1))

            split.append(part)
            total -= part

        return split

    def _draw_with_mean(self, low: int, high: int, mean: float) -> int:
        """Draw an integer in [low, high] from a distribution with the given mean.

        low + (high - low) * U ** (1 / shape) has mean low + (high - low) * shape /
        (shape + 1), so solving for the shape places the mean anywhere in the range --
        which a fixed-shape distribution cannot do once the target nears a bound.
        """
        span = high - low

        if span <= 0:
            return low

        target = (mean - low) / span

        if target <= 0:
            return low

        if target >= 1:
            return high

        shape = target / (1 - target)
        part = low + span * self._rng.random() ** (1 / shape)

        return min(max(round(part), low), high)

    def _coin_flips(self, size: int) -> int:
        return sum(1 for _ in range(size) if self._rng.random() < 0.5)

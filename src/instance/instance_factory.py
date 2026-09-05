from src.instance.instance import Instance
from typing import List, Optional
import os
import random

# a zero-sum component needs one payer and one receiver at the very least
MIN_COMPONENT_SIZE = 2

class InstanceFactory:
    """Draws balance vectors with a planted partition into K zero-sum components.

    The optimum of the problem is N - k*, where k* is the largest number of parts in a
    partition of the balances into zero-sum subsets. Drawing balances at random leaves k*
    to chance, so instead the vector is assembled one component at a time, which makes k*
    at least K and so puts a known ceiling on the optimum. Accidental zero-sum subsets can
    only push k* higher, so K is a lower bound rather than an exact figure -- and their
    number grows like 2^N / (2 * N * B), so keeping a low K meaningful takes a B that
    scales with N.

    Components are of near-equal size by default, which makes (N, K) a complete statement
    about the structure: every component holds N // K people, give or take the remainder.
    Pass component_sizes instead to shape them by hand.
    """

    def __init__(
        self,
        N: int,
        B: int,
        K: Optional[int] = None,
        component_sizes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if N < 2:
            raise ValueError('N must be at least 2')

        if B < 1:
            raise ValueError('B must be at least 1')

        if (K is None) == (component_sizes is None):
            raise ValueError('pass exactly one of K and component_sizes')

        if K is not None and not 1 <= K <= N // MIN_COMPONENT_SIZE:
            raise ValueError(f"K must be between 1 and {N // MIN_COMPONENT_SIZE}")

        if component_sizes is not None:
            if not component_sizes:
                raise ValueError('component_sizes must not be empty')

            if any(size < MIN_COMPONENT_SIZE for size in component_sizes):
                raise ValueError(f"every component must hold at least {MIN_COMPONENT_SIZE} people")

            # N is redundant here, which is the point: it catches a typo in the list
            if sum(component_sizes) != N:
                raise ValueError(f"component_sizes must sum to N ({N}), not {sum(component_sizes)}")

        self._N = N
        self._B = B
        self._K = K
        self._component_sizes = component_sizes
        self._rng = random.Random(seed)

    def create(self) -> Instance:
        balances: List[int] = []

        for size in self._sizes():
            balances.extend(self._zero_sum_component(size))

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

    def _sizes(self) -> List[int]:
        """Hand out the N people to the components, as evenly as they divide."""
        if self._component_sizes is not None:
            return self._component_sizes

        size, remainder = divmod(self._N, self._K)

        return [size + 1] * remainder + [size] * (self._K - remainder)

    def _zero_sum_component(self, size: int) -> List[int]:
        """Draw `size` non-zero balances within the range that sum to exactly zero."""
        receivers = self._receiver_count(size)
        payers = size - receivers

        # the component's volume: bounded so that both sides can represent it, since each
        # side splits it into parts of at least 1 and at most B
        volume = self._rng.randint(
            max(receivers, payers),
            min(receivers, payers) * self._B,
        )

        return (
            self._bounded_split(volume, receivers)
            + [-part for part in self._bounded_split(volume, payers)]
        )

    def _receiver_count(self, size: int) -> int:
        """Pick how many of a component's people are receivers, by coin flip within reach.

        Both sides carry the same volume, so a side of k people covers between k and k * B
        -- which leaves the split feasible only where the larger side fits inside what the
        smaller side can reach. That window is the whole of [1, size - 1] unless B is small
        enough to rival the component size.
        """
        span = self._B + 1

        # from size - receivers <= receivers * B and its mirror image
        fewest = max(1, -(-size // span))
        most = min(size - 1, size * self._B // span)

        if fewest > most:
            raise ValueError(
                f"B {self._B} is too small to split a component of {size} people "
                f"into two sides that can settle each other"
            )

        return min(max(self._coin_flips(size), fewest), most)

    def _bounded_split(self, total: int, parts: int) -> List[int]:
        """Split `total` into `parts` integers in [1, B], one at a time.

        Each draw is bounded by what still leaves the remaining parts representable, so the
        split always succeeds as long as parts <= total <= parts * B.

        Drawing uniformly within those bounds would spend the budget at the wrong rate
        whenever the average part is far from B / 2, and the bounds would then pin the whole
        tail of the split to a single value -- which shows up as a pile of balances all
        equal to 1 or all equal to B, exactly the repeats a greedy pairing settles for free.
        So each draw is shaped to have the average still to be spent as its mean, which
        keeps the budget on track and lets no such tail form.
        """
        split: List[int] = []

        for index in range(parts):
            remaining = parts - 1 - index

            low = max(1, total - remaining * self._B)
            high = min(self._B, total - remaining)

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

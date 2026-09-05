from src.instance.instance import Instance
from typing import List, Optional
import os
import numpy as np

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
        self._rng = np.random.default_rng(seed)

    def create(self) -> Instance:
        sizes = np.asarray(self._sizes(), dtype=np.int64)

        # every random draw the instance needs comes out in three calls rather than three
        # per component: numpy's generator costs about the same per call whatever the size,
        # so asking it for ten numbers ten thousand times measured 23x slower than asking
        # once for a hundred thousand
        receivers = self._receiver_counts(sizes)
        payers = sizes - receivers
        volumes = self._volumes(receivers, payers)

        # tolist because the split below is a scalar loop, and np.float64 arithmetic is
        # markedly slower than Python float arithmetic
        uniforms = self._rng.random(self._N).tolist()

        balances: List[int] = []

        for receiver_count, payer_count, volume in zip(receivers, payers, volumes):
            # the offset is passed rather than a slice of uniforms: slicing would copy the
            # tail of the list once per component, which is quadratic in the component count
            taken = len(balances)
            volume = int(volume)
            receiver_count = int(receiver_count)

            balances.extend(self._bounded_split(volume, receiver_count, uniforms, taken))
            balances.extend(
                -part for part in
                self._bounded_split(volume, int(payer_count), uniforms, taken + receiver_count)
            )

        drawn = np.array(balances, dtype=np.int64)

        # the components are contiguous as built, which would hand the partition away
        self._rng.shuffle(drawn)

        return Instance(drawn)

    def create_as_txt(self, file_path: str) -> None:
        instance = self.create()

        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        # np.savetxt formats row by row in Python and measured 6.7x slower than this
        with open(file_path, 'w') as file:
            file.write('\n'.join(map(str, instance.balances.tolist())))
            file.write('\n')

    def _sizes(self) -> List[int]:
        """Hand out the N people to the components, as evenly as they divide."""
        if self._component_sizes is not None:
            return self._component_sizes

        size, remainder = divmod(self._N, self._K)

        return [size + 1] * remainder + [size] * (self._K - remainder)

    def _receiver_counts(self, sizes: np.ndarray) -> np.ndarray:
        """Pick how many of each component's people are receivers, by coin flip within reach.

        Both sides carry the same volume, so a side of k people covers between k and k * B
        -- which leaves the split feasible only where the larger side fits inside what the
        smaller side can reach. That window is the whole of [1, size - 1] unless B is small
        enough to rival the component size.
        """
        span = self._B + 1

        # from size - receivers <= receivers * B and its mirror image
        fewest = np.maximum(1, -(-sizes // span))
        most = np.minimum(sizes - 1, sizes * self._B // span)

        unreachable = fewest > most

        if unreachable.any():
            size = int(sizes[np.argmax(unreachable)])

            raise ValueError(
                f"B {self._B} is too small to split a component of {size} people "
                f"into two sides that can settle each other"
            )

        return np.clip(self._rng.binomial(sizes, 0.5), fewest, most)

    def _volumes(self, receivers: np.ndarray, payers: np.ndarray) -> np.ndarray:
        """Draw each component's volume, bounded so that both of its sides can represent it.

        Each side splits the volume into parts of at least 1 and at most B, so the volume
        has to fit between what the larger side needs and what the smaller side can reach.
        """
        return self._rng.integers(
            np.maximum(receivers, payers),
            np.minimum(receivers, payers) * self._B,
            endpoint=True,
        )

    def _bounded_split(self, total: int, parts: int, uniforms: List[float], offset: int) -> List[int]:
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

        # the loop cannot be vectorised -- every bound depends on what the earlier draws
        # spent -- so it consumes randomness that create() already drew in bulk
        for index in range(parts):
            remaining = parts - 1 - index

            low = max(1, total - remaining * self._B)
            high = min(self._B, total - remaining)

            part = self._draw_with_mean(low, high, total / (remaining + 1), uniforms[offset + index])

            split.append(part)
            total -= part

        return split

    @staticmethod
    def _draw_with_mean(low: int, high: int, mean: float, uniform: float) -> int:
        """Turn a uniform draw into an integer in [low, high] whose distribution has the given mean.

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
        part = low + span * uniform ** (1 / shape)

        return int(min(max(round(part), low), high))

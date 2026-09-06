from typing import Optional, Tuple
import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.permutation_decoder import PermutationDecoder
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.instance.instance import Instance

class RandomKeyAlgorithm(BaseAlgorithm):
    """Builds a solution from a vector of random keys, one per person.

    The keys order the people, and the decoder turns that order into a settlement. All
    this class decides is the order, which is the whole of what a random-key encoding
    contributes.

    Written as a BRKGA decoder, so the object is reusable: set_random_keys swaps the
    chromosome without rebuilding anything, and one PermutationDecoder is shared across
    every call.

    The chromosome is optional, because it only makes sense once the instance is known --
    it needs exactly one key per person. Without one, run draws its own for whatever
    instance it is handed, which is what lets a single object serve a whole benchmark.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._decoder = PermutationDecoder()
        self._rng = np.random.default_rng(seed)
        self._random_keys: Optional[np.ndarray] = None

    def name(self) -> str:
        return 'random_key'

    @property
    def random_keys(self) -> Optional[np.ndarray]:
        return self._random_keys

    def set_random_keys(self, random_keys: Optional[np.ndarray]) -> None:
        if random_keys is None:
            self._random_keys = None

            return

        keys = np.asarray(random_keys, dtype=np.float64)

        if keys.ndim != 1:
            raise ValueError('random_keys must be one-dimensional')

        if keys.size and (keys.min() < 0.0 or keys.max() > 1.0):
            raise ValueError('random_keys must lie between 0 and 1')

        # no length check on purpose: swapping in a chromosome of another length is how
        # one object serves instances of different sizes
        self._random_keys = keys

    def supports(self, instance: Instance) -> bool:
        # without a chromosome any instance is fair game, since run draws one to fit
        return self._random_keys is None or len(self._random_keys) == len(instance.balances)

    def run(self, instance: Instance) -> RunResult:
        _, solution = self.construct(instance)

        return RunResult(solution=solution)

    def construct(self, instance: Instance) -> Tuple[np.ndarray, Solution]:
        """The order the keys imply, and the solution it decodes to.

        Handing back the order as well as the solution is what lets a local search pick up
        where this leaves off: the solution alone would say how good the starting point is
        but not what to perturb.
        """
        keys = self._keys_for(instance)
        left = self._decoder.survivors(instance.balances)

        # the key order restricted to whoever is left, already in instance indices, so the
        # groups come back global and need no remapping. Stable so that equal keys keep
        # person order, which keeps a run reproducible
        order = left[np.argsort(keys[left], kind='stable')]

        return order, self._decoder.decode(instance, order)

    def _keys_for(self, instance: Instance) -> np.ndarray:
        """The chromosome to decode this instance with, drawn on the spot if there is none.

        A drawn chromosome stays local to the call and is never kept: holding on to it
        would leave the object sized to whichever instance came first, and every instance
        of another size would stop being supported from then on.
        """
        people = len(instance.balances)

        if self._random_keys is None:
            return self._rng.random(people)

        if len(self._random_keys) != people:
            raise ValueError(
                f"{len(self._random_keys)} random keys cannot decode an instance of "
                f"{people} people"
            )

        return self._random_keys

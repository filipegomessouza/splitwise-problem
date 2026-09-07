from typing import Optional, Tuple
import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.permutation_decoder import PermutationDecoder
from src.algorithms.random_keys import RandomKeys
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
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._decoder = PermutationDecoder()
        self._keys = RandomKeys(seed)

    def name(self) -> str:
        return 'random_key'

    @property
    def random_keys(self) -> Optional[np.ndarray]:
        return self._keys.keys

    def set_random_keys(self, random_keys: Optional[np.ndarray]) -> None:
        self._keys.set(random_keys)

    def supports(self, instance: Instance) -> bool:
        return self._keys.fits(len(instance.balances))

    def run(self, instance: Instance) -> RunResult:
        _, solution = self.construct(instance)

        return RunResult(solution=solution)

    def construct(self, instance: Instance) -> Tuple[np.ndarray, Solution]:
        """The order the keys imply, and the solution it decodes to.

        Handing back the order as well as the solution is what lets a local search pick up
        where this leaves off: the solution alone would say how good the starting point is
        but not what to perturb.

        Only the survivors are ordered. The people the pairing already settled are settled
        the same way whatever the order says, so ordering them decides nothing.
        """
        balances = instance.balances
        order = self._keys.order(len(balances), self._decoder.survivors(balances))

        return order, self._decoder.decode(instance, order)

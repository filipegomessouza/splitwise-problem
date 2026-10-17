from typing import Optional, Sequence, Tuple
import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.order_evaluator import OrderEvaluator
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
        self._instance: Optional[Instance] = None
        self._evaluator: Optional[OrderEvaluator] = None
        self._survivors: Optional[np.ndarray] = None

    def name(self) -> str:
        return 'random_key'

    @property
    def random_keys(self) -> Optional[np.ndarray]:
        return self._keys.keys

    def set_random_keys(self, random_keys: Optional[np.ndarray]) -> None:
        self._keys.set(random_keys)

    def set_instance(self, instance: Optional[Instance]) -> None:
        """The instance decode scores.

        Needed because the BRKGA library calls decode with a chromosome and nothing else,
        so the instance has to be known beforehand. run and construct take theirs as an
        argument and ignore this one.

        Whatever decode needs that depends on the instance alone -- the survivors to
        order, and the evaluator that scores an order -- is worked out here, once, rather
        than once per chromosome.
        """
        self._instance = instance

        if instance is None:
            self._evaluator = None
            self._survivors = None

            return

        self._evaluator = OrderEvaluator(instance)
        self._survivors = self._decoder.survivors(instance.balances)

    def decode(self, chromosome: Sequence[float], rewrite: bool) -> float:
        """The fitness a chromosome decodes to, as the BRKGA library expects it.

        The chromosome is taken as this object's keys and ordered exactly as construct
        orders them, but only scored: a BRKGA reads one number per chromosome, so the
        Solution construct would build is never made. OrderEvaluator.fitness counts what
        PermutationDecoder.decode would settle, so the two cannot disagree about what a
        chromosome is worth.

        `rewrite` is ignored: decoding never changes the chromosome, since no local search
        runs here to produce a better order worth writing back.
        """
        if self._instance is None:
            raise ValueError('set_instance must be called before decode')

        self.set_random_keys(np.asarray(chromosome, dtype=np.float64))
        order = self._keys.order(len(self._instance.balances), self._survivors)

        return float(self._evaluator.fitness(order))

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

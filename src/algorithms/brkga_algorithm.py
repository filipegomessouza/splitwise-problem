import math
from typing import List, Optional, Tuple
import numpy as np
from brkga_mp_ipr.enums import Sense
from brkga_mp_ipr.types import BrkgaParams
from src.algorithms.base_algorithm import METRICS, BaseAlgorithm
from src.algorithms.elitist_brkga_mp_ipr import ElitistBrkgaMpIpr
from src.algorithms.permutation_decoder import PermutationDecoder
from src.algorithms.random_key_algorithm import RandomKeyAlgorithm
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.instance.instance import Instance

class BrkgaAlgorithm(BaseAlgorithm):
    """A biased random-key genetic algorithm over the random-key encoding.

    A thin shell around brkga_mp_ipr: the library evolves the chromosomes (through
    ElitistBrkgaMpIpr, so that its elite is really kept), and RandomKeyAlgorithm decodes
    them, so a chromosome here means exactly what it means to the random-key constructor
    -- one key per person, ordering the survivors.

    Evolution stops at whichever comes first of the criteria given: max_generations,
    time_limit seconds of evolving, or max_stalled_generations in a row without the best
    fitness improving. Each is optional, but at least one is required.

    The population is population_size, 100 by default. With population_size=None it grows
    with the instance instead, as the BRKGA literature recommends: population_factor times
    the number of keys the decoder reads -- the survivors of the pairing, not the whole
    chromosome, since the other keys never change a decode -- rounded up. Over a fixed
    number of generations that did worse here, and took about 7 times as long.

    Mating defaults to the classic BRKGA: one elite and one non-elite parent, with each
    gene taken from the better of the two with elite_inheritance_probability. None of the
    library's bias functions gives a chosen probability, so it is set as a custom one:
    the best parent weighs elite_inheritance_probability, and the rest share what is left.

    Also a PermutationConstructor, so a local search can take the best chromosome's order
    as its starting point: FirstImprovementAlgorithm(BrkgaAlgorithm()) is a BRKGA followed
    by a first-improvement search on its best individual.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        population_size: Optional[int] = 100,
        population_factor: float = 1.0,
        elite_percentage: float = 0.20,
        mutants_percentage: float = 0.15,
        total_parents: int = 2,
        num_elite_parents: int = 1,
        elite_inheritance_probability: float = 0.7,
        num_independent_populations: int = 1,
        max_generations: Optional[int] = 1000,
        time_limit: Optional[float] = None,
        max_stalled_generations: Optional[int] = None,
    ) -> None:
        # checked here and not only by evolve: the runner reports a failed run and moves
        # on, so a configuration that can never run would fail once per instance
        if max_generations is None and time_limit is None and max_stalled_generations is None:
            raise ValueError(
                'BrkgaAlgorithm needs max_generations, time_limit or max_stalled_generations'
            )

        if population_factor < 1.0:
            raise ValueError('population_factor must be at least 1')

        # plain values only, and the library object is built in run: it holds its bias
        # function as a lambda, which the runner could neither deep-copy nor pickle
        self._seed = seed
        self._population_size = population_size
        self._population_factor = population_factor
        self._elite_percentage = elite_percentage
        self._mutants_percentage = mutants_percentage
        self._total_parents = total_parents
        self._num_elite_parents = num_elite_parents
        self._elite_inheritance_probability = elite_inheritance_probability
        self._num_independent_populations = num_independent_populations
        self._max_generations = max_generations
        self._time_limit = time_limit
        self._max_stalled_generations = max_stalled_generations

    def name(self) -> str:
        return 'brkga'

    def metrics(self) -> List[str]:
        return METRICS + ['iterations']

    def run(self, instance: Instance) -> RunResult:
        """Evolve, then decode the best chromosome; iterations counts the generations."""
        _, solution, generations = self._evolve(instance)

        return RunResult(solution=solution, iterations=generations)

    def construct(self, instance: Instance) -> Tuple[np.ndarray, Solution]:
        """The best chromosome's order of the survivors, and the solution it decodes to."""
        order, solution, _ = self._evolve(instance)

        return order, solution

    def _evolve(self, instance: Instance) -> Tuple[np.ndarray, Solution, int]:
        """The best chromosome's order and solution, and how many generations it took.

        Chromosomes are only scored while evolving; the one Solution built is the best's.
        """
        decoder = RandomKeyAlgorithm()
        decoder.set_instance(instance)

        brkga = ElitistBrkgaMpIpr(
            decoder=decoder,
            sense=Sense.MINIMIZE,
            seed=self._seed,
            # one key per person, as RandomKeys requires: the keys of people the pairing
            # already settled are carried along but never read
            chromosome_size=len(instance.balances),
            params=self._params(self._population_for(instance)),
        )
        brkga.set_bias_custom_function(self._bias)
        brkga.initialize()
        generations = brkga.evolve(
            num_generations=self._max_generations,
            time_limit=self._time_limit,
            max_stalled_generations=self._max_stalled_generations,
        )

        decoder.set_random_keys(np.asarray(brkga.get_best_chromosome(), dtype=np.float64))
        order, solution = decoder.construct(instance)

        return order, solution, generations

    def _bias(self, rank: int) -> float:
        """Weight of the parent ranked `rank` among those mating, 1 being the best."""
        if rank == 1:
            return self._elite_inheritance_probability

        return (1.0 - self._elite_inheritance_probability) / (self._total_parents - 1)

    def _population_for(self, instance: Instance) -> int:
        if self._population_size is not None:
            return self._population_size

        survivors = PermutationDecoder().survivors(instance.balances)

        return math.ceil(self._population_factor * len(survivors))

    def _params(self, population_size: int) -> BrkgaParams:
        params = BrkgaParams()
        params.population_size = population_size
        params.elite_percentage = self._elite_percentage
        params.mutants_percentage = self._mutants_percentage
        params.total_parents = self._total_parents
        params.num_elite_parents = self._num_elite_parents
        params.num_independent_populations = self._num_independent_populations

        return params

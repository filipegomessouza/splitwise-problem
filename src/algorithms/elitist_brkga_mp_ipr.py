import time
from typing import Optional
from brkga_mp_ipr.algorithm import BrkgaMpIpr
from brkga_mp_ipr.enums import Sense
from brkga_mp_ipr.types import Population

class ElitistBrkgaMpIpr(BrkgaMpIpr):
    """BrkgaMpIpr with its elite actually carried over, and more ways to stop evolving.

    brkga_mp_ipr 0.9.1 copies the elite by list position (algorithm.py, in
    evolve_population), while it keeps fitness sorted apart from the chromosomes. Left
    alone, the elite it keeps is not the best chromosomes, and get_best_chromosome returns
    one that decodes worse than get_best_fitness reports.

    Every population is put back in fitness order whenever the library has just sorted
    it, so that position i always holds the i-th best chromosome and the library's own
    elite copy becomes correct. initialize covers reset too, which goes through it, and
    evolve_population covers evolve, which calls it once per population.

    The library's evolve only runs a fixed number of generations. Here it also stops on a
    time limit or after a number of generations without the best fitness improving --
    whichever criterion given is met first.
    """

    def initialize(self) -> None:
        super().initialize()

        for population in self._current_populations:
            self._align(population)

    def evolve(
        self,
        num_generations: Optional[int] = None,
        time_limit: Optional[float] = None,
        max_stalled_generations: Optional[int] = None,
    ) -> int:
        """Evolve until one of the given criteria is met, and return the generations run.

        Every criterion is optional, but at least one is required, or nothing would stop
        it. time_limit counts seconds from this call and is checked between generations,
        so it can be overshot by up to one generation. A generation stalls when it leaves
        the best fitness no better than before, ties included.
        """
        if num_generations is None and time_limit is None and max_stalled_generations is None:
            raise ValueError(
                'evolve needs num_generations, time_limit or max_stalled_generations'
            )

        started = time.perf_counter()
        best = self.get_best_fitness()
        generations = 0
        stalled = 0

        while (
            (num_generations is None or generations < num_generations)
            and (time_limit is None or time.perf_counter() - started < time_limit)
            and (max_stalled_generations is None or stalled < max_stalled_generations)
        ):
            super().evolve(1)
            generations += 1

            fitness = self.get_best_fitness()

            if self._improves(fitness, best):
                best = fitness
                stalled = 0
            else:
                stalled += 1

        return generations

    def evolve_population(self, population_index: int) -> None:
        super().evolve_population(population_index)

        # the library swaps the new generation into the current slot on its way out
        self._align(self._current_populations[population_index])

    def _improves(self, fitness: float, best: float) -> bool:
        if self.opt_sense == Sense.MINIMIZE:
            return fitness < best

        return fitness > best

    def _align(self, population: Population) -> None:
        """Put the chromosomes in fitness order, so that fitness[i] = (value, i).

        Reaches into the population because nothing public reorders it.
        """
        population.chromosomes = [population.chromosomes[index] for _, index in population.fitness]
        population.fitness = [(value, i) for i, (value, _) in enumerate(population.fitness)]

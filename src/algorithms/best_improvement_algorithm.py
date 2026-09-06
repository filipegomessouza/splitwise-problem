from typing import Optional
import numpy as np
from src.algorithms.local_search_algorithm import LocalSearchAlgorithm, Step
from src.algorithms.solution import Solution

class BestImprovementAlgorithm(LocalSearchAlgorithm):
    """Moves to the best neighbour in the whole swap neighbourhood, or stops.

    Every step costs the full O(n^2) neighbourhood, against first-improvement's expected
    fraction of it, and buys the steepest descent available -- fewer, better steps for more
    work per step. Which of the two wins is an empirical question on these instances, and
    the point of keeping both under the same base.
    """

    def name(self) -> str:
        return 'best_improvement'

    def _accept(self, order: np.ndarray, solution: Solution) -> Optional[Step]:
        instance = solution.instance
        positions = len(order)

        best: Optional[Step] = None
        # seeded with the incumbent, so a neighbour has to be strictly better to be kept
        # at all, and ties among equally good neighbours go to the first one scanned
        best_fitness = solution.fitness

        for i in range(positions - 1):
            for j in range(i + 1, positions):
                candidate, candidate_solution = self._neighbour(instance, order, i, j)

                if candidate_solution.fitness < best_fitness:
                    best_fitness = candidate_solution.fitness
                    best = (candidate, candidate_solution)

        return best

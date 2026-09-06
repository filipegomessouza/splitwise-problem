from typing import Optional
import numpy as np
from src.algorithms.local_search_algorithm import LocalSearchAlgorithm, Step
from src.algorithms.solution import Solution

class FirstImprovementAlgorithm(LocalSearchAlgorithm):
    """Moves to the first neighbour that improves on the incumbent, or stops.

    Only a local optimum costs the full O(n^2) neighbourhood; every step that finds
    something stops early, so a step is cheap where gains are plentiful and grows expensive
    only as the order runs out of them. The trade against best-improvement is more steps of
    less depth for far less work each.

    Every step rescans from the first pair rather than resuming where the last one stopped.
    The swap that was accepted changed the order, so the pairs already rejected are not the
    same pairs any more, and skipping them would skip neighbours that were never seen.
    """

    def name(self) -> str:
        return 'first_improvement'

    def _accept(self, order: np.ndarray, solution: Solution) -> Optional[Step]:
        instance = solution.instance
        positions = len(order)

        for i in range(positions - 1):
            for j in range(i + 1, positions):
                candidate, candidate_solution = self._neighbour(instance, order, i, j)

                if candidate_solution.fitness < solution.fitness:
                    return candidate, candidate_solution

        return None

from typing import List, Optional, Tuple
from src.algorithms.local_search_algorithm import LocalSearchAlgorithm, Step
from src.algorithms.order_evaluator import OrderEvaluator

class BestImprovementAlgorithm(LocalSearchAlgorithm):
    """Moves to the best neighbour in the whole swap neighbourhood, or stops.

    Every step costs the full O(n^2) neighbourhood, against first-improvement's expected
    fraction of it, and buys the steepest descent available -- fewer, better steps for more
    work per step. Which of the two wins is an empirical question on these instances, and
    the point of keeping both under the same base.
    """

    def suffix(self) -> str:
        return 'bi'

    def _accept(
        self,
        evaluator: OrderEvaluator,
        order: List[int],
        fitness: int,
    ) -> Optional[Step]:
        positions = len(order)
        prefixes = evaluator.prefixes(order)
        balances = evaluator.balances

        best: Optional[Tuple[int, int]] = None
        # seeded with the incumbent, so a neighbour has to be strictly better to be kept
        # at all, and ties among equally good neighbours go to the first one scanned
        best_fitness = fitness

        for i in range(positions - 1):
            prefix = prefixes[i]
            balance = balances[order[i]]

            for j in range(i + 1, positions):
                if balances[order[j]] == balance:
                    continue

                candidate_fitness = evaluator.fitness_from(prefix, order, i, j)

                if candidate_fitness < best_fitness:
                    best_fitness = candidate_fitness
                    best = (i, j)

        if best is None:
            return None

        # the winner is swapped for real only now: every neighbour up to here was scored
        # off a copy of the tail, never off a whole rebuilt order
        return self._swap(order, *best), best_fitness

from typing import List, Optional
from src.algorithms.local_search_algorithm import LocalSearchAlgorithm, Step
from src.algorithms.order_evaluator import OrderEvaluator

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

    def suffix(self) -> str:
        return 'fi'

    def _accept(
        self,
        evaluator: OrderEvaluator,
        order: List[int],
        fitness: int,
    ) -> Optional[Step]:
        positions = len(order)

        # built for every position even when the scan returns from the first one: it is one
        # O(n) pass against the O(n^2) neighbourhood it serves
        prefixes = evaluator.prefixes(order)
        balances = evaluator.balances

        for i in range(positions - 1):
            prefix = prefixes[i]
            balance = balances[order[i]]

            for j in range(i + 1, positions):
                # equal balances swap to the same fitness, so this neighbour cannot win;
                # see OrderEvaluator.balances
                if balances[order[j]] == balance:
                    continue

                candidate_fitness = evaluator.fitness_from(prefix, order, i, j)

                if candidate_fitness < fitness:
                    return self._swap(order, i, j), candidate_fitness

        return None

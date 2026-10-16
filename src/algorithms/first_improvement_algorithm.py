from src.algorithms.local_search_algorithm import LocalSearchAlgorithm

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

    def _stops_at_first(self) -> bool:
        return True

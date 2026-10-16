from src.algorithms.local_search_algorithm import LocalSearchAlgorithm

class BestImprovementAlgorithm(LocalSearchAlgorithm):
    """Moves to the best neighbour in the whole swap neighbourhood, or stops.

    Every step costs the full O(n^2) neighbourhood, against first-improvement's expected
    fraction of it, and buys the steepest descent available -- fewer, better steps for more
    work per step. Which of the two wins is an empirical question on these instances, and
    the point of keeping both under the same base.

    Ties among equally good neighbours go to the first one scanned.
    """

    def suffix(self) -> str:
        return 'bi'

    def _stops_at_first(self) -> bool:
        return False

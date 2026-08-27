from typing import Dict, List, Tuple
import heapq
from src.algorithms.base_algorithm import BaseAlgorithm
from src.instance.instance import Instance


class GreedyAlgorithm(BaseAlgorithm):
    def __init__(self, instance: Instance):
        self._instance = instance

    def run(self) -> int:
        fitness, remaining_values = self.get_values_without_direct_transactions()

        payers = [value for value in remaining_values if value < 0]
        receivers = [-value for value in remaining_values if value > 0]

        heapq.heapify(payers)
        heapq.heapify(receivers)

        while payers and receivers:
            payer = -heapq.heappop(payers)
            receiver = -heapq.heappop(receivers)

            if payer > receiver:
                heapq.heappush(payers, -(payer - receiver))
            elif payer < receiver:
                heapq.heappush(receivers, -(receiver - payer))

            fitness += 1

        return fitness

    def get_values_without_direct_transactions(self) -> Tuple[int, List[int]]:
        counter: Dict[int, int] = {}

        for value in self._instance.values:
            if value != 0:
                counter[value] = counter.get(value, 0) + 1

        direct_transactions = 0

        for value in counter:
            if value > 0 and -value in counter:
                pairs = min(counter[value], counter[-value])
                direct_transactions += pairs

                counter[value] -= pairs
                counter[-value] -= pairs

        remaining_values: List[int] = []

        for value, count in counter.items():
            remaining_values.extend([value] * count)

        return direct_transactions, remaining_values

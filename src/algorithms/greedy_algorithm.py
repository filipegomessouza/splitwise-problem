from typing import List, Dict
import heapq
from src.algorithms.base_algorithm import BaseAlgorithm
from src.instance.instance import Instance


class GreedyAlgorithm(BaseAlgorithm):
    def __init__(self, instance: Instance):
        self._instance = instance
        self._fitness: int = 0

    def run(self) -> int:
        values_without_direct_transactions = self.get_values_without_direct_transactions()

        payers = heapq.heapify([-value for value in values_without_direct_transactions if value < 0])
        receivers = heapq.heapify([value for value in values_without_direct_transactions if value > 0])

        while payers and receivers:
            payer = heapq.heappop(payers)
            receiver = heapq.heappop(receivers)

            if payer > receiver:
                heapq.heappush(payers, payer - receiver)
            elif payer < receiver:
                heapq.heappush(receivers, receiver - payer)

            fitness += 1

        return fitness

    def get_values_without_direct_transactions(self) -> List[int]:
        filtered_values: List[int] = []
        counter: Dict[int, int] = {}

        for value in self._instance.values:
            counter[value] = counter.get(value, 0) + 1

        for value, count in counter.items():
            if value > 0 and -value in counter:
                pairs = min(count, counter[-value])
                self._firness += pairs

                counter[value] -= pairs
                counter[-value] -= pairs

                if counter[value] > 0:
                    filtered_values.extend([value] * counter[value])

                if counter[-value] > 0:
                    filtered_values.extend([-value] * counter[-value])

        return filtered_values

from typing import Dict, List, Tuple
import heapq
from src.algorithms.base_algorithm import BaseAlgorithm
from src.instance.instance import Instance


class GreedyAlgorithm(BaseAlgorithm):
    def __init__(self, instance: Instance):
        self._instance = instance

    def run(self) -> int:
        fitness, remaining_balances = self.get_balances_without_direct_transactions()

        payers = [balance for balance in remaining_balances if balance < 0]
        receivers = [-balance for balance in remaining_balances if balance > 0]

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

    def get_balances_without_direct_transactions(self) -> Tuple[int, List[int]]:
        counter: Dict[int, int] = {}

        for balance in self._instance.balances:
            if balance != 0:
                counter[balance] = counter.get(balance, 0) + 1

        direct_transactions = 0

        for balance in counter:
            if balance > 0 and -balance in counter:
                pairs = min(counter[balance], counter[-balance])
                direct_transactions += pairs

                counter[balance] -= pairs
                counter[-balance] -= pairs

        remaining_balances: List[int] = []

        for balance, count in counter.items():
            remaining_balances.extend([balance] * count)

        return direct_transactions, remaining_balances

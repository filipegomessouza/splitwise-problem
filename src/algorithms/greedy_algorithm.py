from typing import Dict, List, Tuple
import heapq
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.constants.types import TransactionList
from src.instance.instance import Instance

class GreedyAlgorithm(BaseAlgorithm):
    def name(self) -> str:
        return 'greedy'

    def run(self, instance: Instance) -> RunResult:
        transactions, remaining_balances = self.get_balances_without_direct_transactions(instance)

        # heapq is a min-heap, so magnitudes are stored negated to pop the largest first;
        # the person index rides along and breaks ties deterministically
        payers = [(balance, person) for person, balance in remaining_balances if balance < 0]
        receivers = [(-balance, person) for person, balance in remaining_balances if balance > 0]

        heapq.heapify(payers)
        heapq.heapify(receivers)

        while payers and receivers:
            owed, payer = heapq.heappop(payers)
            due, receiver = heapq.heappop(receivers)

            owed = -owed
            due = -due
            amount = min(owed, due)

            transactions.append((payer, receiver, amount))

            if owed > due:
                heapq.heappush(payers, (-(owed - due), payer))
            elif owed < due:
                heapq.heappush(receivers, (-(due - owed), receiver))

        solution = Solution(instance=instance, transactions=transactions)

        return RunResult(solution=solution, status='heuristic')

    def get_balances_without_direct_transactions(
        self, instance: Instance
    ) -> Tuple[TransactionList, List[Tuple[int, int]]]:
        people_by_balance: Dict[int, List[int]] = {}

        for person, balance in enumerate(instance.balances):
            if balance != 0:
                people_by_balance.setdefault(balance, []).append(person)

        transactions: TransactionList = []

        for balance in people_by_balance:
            if balance > 0:
                receivers = people_by_balance[balance]
                payers = people_by_balance.get(-balance)

                if payers is None:
                    continue

                for _ in range(min(len(receivers), len(payers))):
                    transactions.append((payers.pop(), receivers.pop(), balance))

        remaining_balances: List[Tuple[int, int]] = []

        for balance, people in people_by_balance.items():
            remaining_balances.extend((person, balance) for person in people)

        return transactions, remaining_balances

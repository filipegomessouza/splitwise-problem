from dataclasses import dataclass
from src.constants.types import TransactionList
from src.instance.instance import Instance

@dataclass
class Solution:
    """The transactions that settle an instance. Counterpart of Instance."""
    instance: Instance
    transactions: TransactionList

    @property
    def people(self) -> int:
        return len(self.instance.contributions)

    @property
    def fitness(self) -> int:
        return len(self.transactions)

    def describe(self) -> str:
        balances = self.instance.balances

        lines = [
            f"{self.people} people, mean {self.instance.mean}, {self.fitness} transactions",
            '',
            f"{'person':>7} {'contributed':>12} {'balance':>9} {'role':>9}",
        ]

        for person, contribution in enumerate(self.instance.contributions):
            balance = balances[person]
            role = 'payer' if balance < 0 else 'receiver' if balance > 0 else 'settled'
            lines.append(f"{person:>7} {contribution:>12} {balance:>9} {role:>9}")

        lines.append('')
        lines.append('transactions:')

        for payer, receiver, amount in self.transactions:
            lines.append(f"  person {payer} -> person {receiver}: {amount}")

        return '\n'.join(lines)

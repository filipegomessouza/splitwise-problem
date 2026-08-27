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

    def validate(self) -> None:
        """Raise unless the transactions leave everyone having disbursed exactly the mean.

        Checks the settlement, not its cost, so it holds for any algorithm's output.
        """
        net = [0] * self.people

        for payer, receiver, amount in self.transactions:
            if not 0 <= payer < self.people or not 0 <= receiver < self.people:
                raise ValueError(f"transaction ({payer}, {receiver}, {amount}) refers to an unknown person")

            if payer == receiver:
                raise ValueError(f"transaction ({payer}, {receiver}, {amount}) has the same payer and receiver")

            if amount <= 0:
                raise ValueError(f"transaction ({payer}, {receiver}, {amount}) must transfer a positive amount")

            net[payer] -= amount
            net[receiver] += amount

        mean = self.instance.mean

        for person, contribution in enumerate(self.instance.contributions):
            disbursed = contribution - net[person]

            if disbursed != mean:
                raise ValueError(f"person {person} disbursed {disbursed} instead of the mean {mean}")

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

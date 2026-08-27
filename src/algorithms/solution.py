from dataclasses import dataclass
from src.constants.types import TransactionList

@dataclass
class Solution:
    """The transactions that settle an instance. Counterpart of Instance."""
    people: int
    transactions: TransactionList

    @property
    def fitness(self) -> int:
        return len(self.transactions)

    def describe(self) -> str:
        return '\n'.join(
            f"person {payer} -> person {receiver}: {amount}"
            for payer, receiver, amount in self.transactions
        )

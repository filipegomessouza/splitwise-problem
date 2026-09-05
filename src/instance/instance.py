from dataclasses import dataclass
from typing import List

@dataclass
class Instance:
    """How much each person is owed. Negative means the person is a payer, positive a receiver."""
    balances: List[int]

    def __post_init__(self) -> None:
        if sum(self.balances) != 0:
            raise ValueError('balances must sum to zero')

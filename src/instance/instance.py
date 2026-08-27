from dataclasses import dataclass
from typing import List

@dataclass
class Instance:
    """How much each person actually contributed. Balances are derived from the mean."""
    contributions: List[int]

    def __post_init__(self) -> None:
        if any(contribution < 0 for contribution in self.contributions):
            raise ValueError('contributions must be non-negative')

        if self.contributions and sum(self.contributions) % len(self.contributions) != 0:
            raise ValueError('sum of contributions must be a multiple of the number of people')

    @property
    def mean(self) -> int:
        if not self.contributions:
            return 0

        return sum(self.contributions) // len(self.contributions)

    @property
    def balances(self) -> List[int]:
        """Negative means the person is a payer, positive means a receiver."""
        mean = self.mean

        return [contribution - mean for contribution in self.contributions]

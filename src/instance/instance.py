from dataclasses import dataclass
import numpy as np

@dataclass
class Instance:
    """How much each person is owed. Negative means the person is a payer, positive a receiver."""
    balances: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        self.balances = np.asarray(self.balances, dtype=np.int64)

        if self.balances.sum() != 0:
            raise ValueError('balances must sum to zero')

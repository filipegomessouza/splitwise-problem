from src.instance.instance import Instance
from typing import List, Optional
import os
import random

class InstanceFactory:
    def __init__(self, max_value: int, size: int, seed: Optional[int] = None):
        if max_value < 0:
            raise ValueError('max_value must be non-negative')

        if size < 2:
            raise ValueError('size must be at least 2')

        self._max_value = max_value
        self._size = size
        self._rng = random.Random(seed)

    def create(self) -> Instance:
        values = [self._rng.randint(-self._max_value, self._max_value) for _ in range(self._size)]
        self._balance(values)

        return Instance(values)

    def create_as_txt(self, file_path: str) -> None:
        instance = self.create()

        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, 'w') as file:
            for value in instance.values:
                file.write(f"{value}\n")

    def _balance(self, values: List[int]) -> None:
        """Spread the residual over the values so the sum is 0 and every value stays in range."""
        residual = sum(values)
        indices = list(range(len(values)))

        while residual != 0:
            self._rng.shuffle(indices)
            share = max(1, abs(residual) // len(indices))

            for index in indices:
                if residual == 0:
                    break

                if residual > 0:
                    delta = min(share, residual, values[index] + self._max_value)
                    values[index] -= delta
                    residual -= delta
                else:
                    delta = min(share, -residual, self._max_value - values[index])
                    values[index] += delta
                    residual += delta

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
        contributions = [self._rng.randint(0, self._max_value) for _ in range(self._size)]
        self._align_mean(contributions)

        return Instance(contributions)

    def create_as_txt(self, file_path: str) -> None:
        instance = self.create()

        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, 'w') as file:
            for contribution in instance.contributions:
                file.write(f"{contribution}\n")

    def _align_mean(self, contributions: List[int]) -> None:
        residual = sum(contributions) % len(contributions)
        indices = list(range(len(contributions)))

        while residual != 0:
            self._rng.shuffle(indices)
            share = max(1, residual // len(indices))

            for index in indices:
                if residual == 0:
                    break

                delta = min(share, residual, contributions[index])
                contributions[index] -= delta
                residual -= delta

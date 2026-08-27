from src.instance.instance import Instance
from typing import Optional
import random

class InstanceFactory:
    def __init__(self, min_value: int, max_value: int, size: int, seed: Optional[int] = None):
        self._min_value = min_value
        self._max_value = max_value
        self._size = size
        self._rng = random.Random(seed)

    def create(self) -> Instance:
        return Instance([self._rng.randint(self._min_value, self._max_value) for _ in range(self._size)])

    def create_as_txt(self, file_path: str) -> None:
        instance = self.create()

        with open(file_path, 'w') as file:
            for value in instance.values:
                file.write(f"{value}\n")

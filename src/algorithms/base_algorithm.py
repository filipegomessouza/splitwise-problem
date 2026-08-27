from abc import ABC, abstractmethod
from src.algorithms.solution import Solution

class BaseAlgorithm(ABC):
    @abstractmethod
    def run(self) -> Solution:
        pass

from abc import ABC, abstractmethod
from src.algorithms.run_result import RunResult
from src.instance.instance import Instance

class BaseAlgorithm(ABC):
    @abstractmethod
    def name(self) -> str:
        """Short identifier used to label this algorithm's columns in a report."""
        pass

    @abstractmethod
    def run(self, instance: Instance) -> RunResult:
        pass

    def supports(self, instance: Instance) -> bool:
        """Whether this algorithm can handle the instance at all."""
        return True

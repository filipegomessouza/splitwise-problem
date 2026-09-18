from abc import ABC, abstractmethod
from typing import List
from src.algorithms.run_result import RunResult
from src.instance.instance import Instance

# measured for every algorithm, so each one contributes at least this block of columns
METRICS = [
    'fitness',
    'seconds',
]

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

    def metrics(self) -> List[str]:
        """The column suffixes this algorithm contributes to a report.

        Overridden by algorithms that report something the others cannot, so that a
        column shows up only where it means anything.
        """
        return METRICS

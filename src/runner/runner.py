import time
from typing import Any, Dict, List, Optional
import pandas as pd
from src.algorithms.base_algorithm import BaseAlgorithm
from src.instance.instance import Instance

# measured once per algorithm, so each one contributes this block of columns
METRICS = ['fitness', 'seconds', 'status', 'gap', 'valid']

Row = Dict[str, Any]

class Runner:
    """Times every algorithm over every instance, one row per instance."""

    def __init__(self, instances: List[Instance], algorithms: List[BaseAlgorithm]) -> None:
        self._instances = instances
        self._algorithms = algorithms

    def run(self) -> pd.DataFrame:
        rows: List[Row] = []

        for instance in self._instances:
            row: Row = {'people': len(instance.contributions)}

            for algorithm in self._algorithms:
                row.update(self._execute(instance, algorithm))

            rows.append(row)

        return pd.DataFrame(rows, columns=self._columns())

    def _columns(self) -> List[str]:
        return ['people'] + [
            f"{algorithm.name()}_{metric}"
            for algorithm in self._algorithms
            for metric in METRICS
        ]

    def _execute(self, instance: Instance, algorithm: BaseAlgorithm) -> Row:
        name = algorithm.name()

        if not algorithm.supports(instance):
            return self._metrics(name, status='skipped')

        started = time.perf_counter()

        try:
            result = algorithm.run(instance)
        except Exception as error:
            return self._metrics(name, status=f"error: {error}", seconds=time.perf_counter() - started)

        seconds = time.perf_counter() - started

        try:
            result.solution.validate()
            valid = True
        except ValueError:
            valid = False

        return self._metrics(
            name,
            status=result.status,
            seconds=seconds,
            fitness=result.solution.fitness,
            gap=result.gap,
            valid=valid,
        )

    def _metrics(
        self,
        name: str,
        status: str,
        seconds: Optional[float] = None,
        fitness: Optional[int] = None,
        gap: Optional[float] = None,
        valid: Optional[bool] = None,
    ) -> Row:
        return {
            f'{name}_fitness': fitness,
            f'{name}_seconds': seconds,
            f'{name}_status': status,
            f'{name}_gap': gap,
            f'{name}_valid': valid,
        }

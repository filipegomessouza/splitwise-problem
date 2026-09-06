import sys
import time
from typing import Any, Dict, List, Optional
import pandas as pd
from src.algorithms.base_algorithm import BaseAlgorithm
from src.instance.instance import Instance

# measured once per algorithm, so each one contributes this block of columns
METRICS = ['fitness', 'seconds']

Row = Dict[str, Any]

class Runner:
    """Times every algorithm over every instance, one row per instance."""

    def __init__(self, instances: List[Instance], algorithms: List[BaseAlgorithm]) -> None:
        self._instances = instances
        self._algorithms = algorithms

    def run(self) -> pd.DataFrame:
        rows: List[Row] = []

        for instance in self._instances:
            row: Row = {'people': len(instance.balances)}

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
            return self._metrics(name)

        started = time.perf_counter()

        try:
            result = algorithm.run(instance)
        except Exception as error:
            print(
                f"{name} failed on {len(instance.balances)} people: {error}",
                file=sys.stderr,
            )

            return self._metrics(name, seconds=time.perf_counter() - started)

        seconds = time.perf_counter() - started

        # deliberately outside the try: swallowing this would let a broken algorithm post
        # the best fitness in the table with nothing to flag it
        result.solution.validate()

        return self._metrics(name, seconds=seconds, fitness=result.solution.fitness)

    def _metrics(
        self,
        name: str,
        seconds: Optional[float] = None,
        fitness: Optional[int] = None,
    ) -> Row:
        return {
            f'{name}_fitness': fitness,
            f'{name}_seconds': seconds,
        }

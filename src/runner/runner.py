import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import pandas as pd
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.run_result import RunResult
from src.instance.instance import Instance
from src.helpers.date import now

Row = Dict[str, Any]

class Runner:
    """Times every algorithm over every instance, one row per instance."""

    def run(
        self,
        instances: List[Instance],
        algorithms: List[BaseAlgorithm],
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        rows: List[Row] = []

        print(f'[{now()}] Starting runs')
        print()

        for instance in instances:
            row: Row = {
                'n': len(instance.balances),
                'instance': instance.name,
            }

            for algorithm in algorithms:
                print(f'[{now()}] {instance.name} - {algorithm.name()}')
                row.update(self._execute(instance, algorithm))

            print()
            rows.append(row)
            self._save(rows, output_path)

        print(f'[{now()}] Finished all runs')

        return pd.DataFrame(rows, columns=self._columns(algorithms))

    def _save(self, rows: List[Row], output_path: Optional[str]) -> None:
        if output_path is None:
            return

        directory = os.path.dirname(output_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(output_path, 'w') as file:
            json.dump(rows, file, indent=2)

    def _columns(self, algorithms: List[BaseAlgorithm]) -> List[str]:
        return ['n', 'instance'] + [
            f"{algorithm.name()}_{metric}"
            for algorithm in algorithms
            for metric in algorithm.metrics()
        ]

    def _execute(self, instance: Instance, algorithm: BaseAlgorithm) -> Row:
        name = algorithm.name()

        if not algorithm.supports(instance):
            return self._metrics(algorithm)

        started = time.perf_counter()

        try:
            result = algorithm.run(instance)
        except Exception as error:
            print(
                f"{name} failed on {len(instance.balances)} people: {error}",
                file=sys.stderr,
            )

            return self._metrics(algorithm, seconds=time.perf_counter() - started)

        seconds = time.perf_counter() - started

        # deliberately outside the try: swallowing this would let a broken algorithm post
        # the best fitness in the table with nothing to flag it
        result.solution.validate()

        return self._metrics(algorithm, seconds=seconds, result=result)

    def _metrics(
        self,
        algorithm: BaseAlgorithm,
        seconds: Optional[float] = None,
        result: Optional[RunResult] = None,
    ) -> Row:
        measured = {
            'fitness': result.solution.fitness if result is not None else None,
            'seconds': seconds,
            'proven': result.proven if result is not None else None,
            'iterations': result.iterations if result is not None else None,
        }

        # an algorithm that declares a metric nobody measures fails loudly here, rather
        # than quietly contributing an all-empty column
        return {
            f'{algorithm.name()}_{metric}': measured[metric]
            for metric in algorithm.metrics()
        }

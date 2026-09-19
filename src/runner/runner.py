import copy
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
import pandas as pd
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.run_result import RunResult
from src.instance.instance import Instance
from src.helpers.date import now

Row = Dict[str, Any]

# spawn rather than the platform default: fork copies a process that numpy and gurobipy may
# have left threads in, and Linux switches its default to forkserver in 3.14 anyway. Every
# method but fork needs an `if __name__ == '__main__':` guard in the launching script, so
# naming one here is what keeps a run behaving the same wherever it happens
START_METHOD = 'spawn'

class Task(NamedTuple):
    """One algorithm on one instance: the unit of work handed to a worker.

    `index` is the position of the instance in the caller's list, carried along so a result
    can find its row again after the pool has reordered everything by completion time.
    """
    index: int
    instance: Instance
    algorithm: BaseAlgorithm

class Runner:
    """Times every algorithm over every instance, one row per instance.

    Every (instance, algorithm) pair is a task of its own rather than every instance, which
    is what keeps the tail short: the local searches cost O(n^3) per iteration, so the
    largest instance decides when the whole benchmark ends, and splitting it across workers
    is the only way to overlap its algorithms with anything.
    """

    def run(
        self,
        instances: List[Instance],
        algorithms: List[BaseAlgorithm],
        output_path: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        """Measure every algorithm on every instance, `max_workers` runs at a time.

        `max_workers` caps how many (instance, algorithm) runs execute at once; None takes
        every cpu this process is allowed to use. At 1 nothing is forked at all, and the
        numbers come out identical to any other setting -- see `_execute`.
        """
        workers = self._workers(max_workers, len(instances) * len(algorithms))
        tasks = self._tasks(instances, algorithms, ordered=workers == 1)

        rows = [self._skeleton(instance, algorithms) for instance in instances]

        # what tells a metric that has not been measured yet from one measured as None: both
        # read as None in the skeleton, and only the first must be kept out of a checkpoint
        remaining = [len(algorithms)] * len(instances)

        print(f'[{now()}] Starting {len(tasks)} runs on {workers} worker(s)')
        print()

        started = time.perf_counter()

        for done, (task, measured) in enumerate(self._results(tasks, workers), start=1):
            rows[task.index].update(measured)
            remaining[task.index] -= 1

            print(self._progress(task, measured, done, len(tasks)))

            # a checkpoint only once an instance is whole: a half-measured row reads like a
            # real result to anything that opens the file, and rewriting the same completed
            # rows after every single task would buy nothing
            if remaining[task.index] == 0:
                self._save(self._finished(rows, remaining), output_path)

        print()
        print(f'[{now()}] Finished all runs in {time.perf_counter() - started:.1f}s')

        return pd.DataFrame(rows, columns=self._columns(algorithms))

    def _workers(self, max_workers: Optional[int], tasks: int) -> int:
        if max_workers is not None:
            if max_workers < 1:
                raise ValueError('max_workers must be at least 1')

            return max(1, min(max_workers, tasks))

        # sched_getaffinity rather than cpu_count: under taskset or a cpu-limited container
        # the latter reports the machine's cores instead of the ones this process may use
        available = (
            len(os.sched_getaffinity(0))
            if hasattr(os, 'sched_getaffinity')
            else os.cpu_count() or 1
        )

        return max(1, min(available, tasks))

    def _tasks(
        self,
        instances: List[Instance],
        algorithms: List[BaseAlgorithm],
        ordered: bool,
    ) -> List[Task]:
        tasks = [
            Task(index, instance, algorithm)
            for index, instance in enumerate(instances)
            for algorithm in algorithms
        ]

        if ordered:
            return tasks

        # longest first: with a bounded pool the submission order sets the makespan, and one
        # that starts the 1500-person searches last has nothing cheap left to overlap them
        # with. Safe only because every run draws its own keys -- see _execute -- so the
        # order tasks go out in no longer reaches the numbers that come back
        return sorted(tasks, key=lambda task: -len(task.instance.balances))

    def _results(self, tasks: List[Task], workers: int) -> Iterator[Tuple[Task, Row]]:
        """Each task and what it measured, in the order the runs finish."""
        if workers == 1:
            # nothing forked, so a traceback, a breakpoint or a profiler reaches straight
            # into _execute
            for task in tasks:
                yield task, self._execute(task.instance, task.algorithm)

            return

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context(START_METHOD),
        ) as executor:
            futures = {
                executor.submit(self._execute, task.instance, task.algorithm): task
                for task in tasks
            }

            try:
                for future in as_completed(futures):
                    yield futures[future], future.result()
            except BaseException:
                # queued tasks are dropped and the running ones waited out: without the
                # cancel the pool would work through all the remaining runs before letting
                # the failure surface, and without the wait it would leave orphans behind
                executor.shutdown(wait=True, cancel_futures=True)

                raise

    def _skeleton(self, instance: Instance, algorithms: List[BaseAlgorithm]) -> Row:
        """A row with every column present and empty.

        Seeded up front rather than grown as results arrive, because dict.update keeps a
        key where it was first inserted: the saved json then has the same column order
        whatever order the workers happened to finish in.
        """
        row: Row = {column: None for column in self._columns(algorithms)}

        row['n'] = len(instance.balances)
        row['instance'] = instance.name

        return row

    def _finished(self, rows: List[Row], remaining: List[int]) -> List[Row]:
        """The rows whose every algorithm has reported, in the caller's instance order."""
        return [row for row, left in zip(rows, remaining) if left == 0]

    def _progress(self, task: Task, measured: Row, done: int, total: int) -> str:
        """A running count, because once runs interleave it is the only honest signal left.

        The line is printed on completion and carries the result, where the sequential
        runner announced a run before starting it -- a timestamp that would mean nothing
        with several workers going at once.
        """
        name = task.algorithm.name()
        fitness = measured.get(f'{name}_fitness')
        seconds = measured.get(f'{name}_seconds')

        if seconds is None:
            detail = 'unsupported'
        elif fitness is None:
            detail = f'failed after {seconds:.3f}s'
        else:
            detail = f'fitness {fitness} in {seconds:.3f}s'

        return f'[{now()}] ({done}/{total}) {task.instance.name} - {name}: {detail}'

    def _save(self, rows: List[Row], output_path: Optional[str]) -> None:
        if output_path is None:
            return

        directory = os.path.dirname(output_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        # only this process ever writes the file -- workers hand back plain numbers and
        # touch no disk -- so there is no race to guard against, but a checkpoint lands
        # after every instance and a run killed mid-write would leave truncated json where
        # the finished instances were. Written aside and renamed, it is either old or new
        temporary = f'{output_path}.tmp'

        with open(temporary, 'w') as file:
            json.dump(rows, file, indent=2)

        os.replace(temporary, output_path)

    def _columns(self, algorithms: List[BaseAlgorithm]) -> List[str]:
        return ['n', 'instance'] + [
            f"{algorithm.name()}_{metric}"
            for algorithm in algorithms
            for metric in algorithm.metrics()
        ]

    def _execute(self, instance: Instance, algorithm: BaseAlgorithm) -> Row:
        # a copy per run: the constructors carry an rng that advances as it draws, so one
        # shared object makes an instance's result depend on how many instances happened to
        # run before it. Crossing a process boundary copies anyway; doing it here as well is
        # what keeps max_workers=1 measuring the very same numbers as max_workers=16
        algorithm = copy.deepcopy(algorithm)
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
        # the best fitness in the table with nothing to flag it. Raised in a worker it comes
        # back through the future and is re-raised here, run and all, as it always was
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

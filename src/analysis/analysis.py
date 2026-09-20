import os
from typing import Any, Callable, Dict, List, Optional
import matplotlib
import pandas as pd

# nothing here opens a window, and importing an interactive backend would fail outright
# on a machine without a display
matplotlib.use('Agg')

import matplotlib.pyplot as plt

Row = Dict[str, Any]

# series that land on the same point stay distinguishable: the local searches routinely
# tie on fitness, and identical markers would hide one behind the other
MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')

class Analysis:
    """Charts a runner's rows, one point per instance per column."""

    def plot_results(
        self,
        rows: List[Row],
        columns: Dict[str, str],
        title: str,
        file_path: str,
        y_label: Optional[str] = None,
        log_scale: bool = False,
        format_instance: Optional[Callable[[str], str]] = None,
    ) -> str:
        """Scatter every column against the instances and return the path written.

        The x axis is categorical -- one slot per instance, merely *ordered* by size --
        because distinct instances share the same n, and a numeric axis would stack them
        on top of each other. Instance names are labelled through format_instance so that
        this class stays ignorant of how any particular instance set names its files.
        """
        ordered = self._ordered(rows)
        positions = range(len(ordered))
        labels = [self._label(row, format_instance) for row in ordered]

        figure, axes = plt.subplots(figsize=(10, 6))

        for index, (column, legend) in enumerate(columns.items()):
            axes.scatter(
                positions,
                self._values(ordered, column),
                label=legend,
                marker=MARKERS[index % len(MARKERS)],
                alpha=0.8,
            )

        axes.set_title(title)
        axes.set_xlabel('Instance')

        if y_label is not None:
            axes.set_ylabel(y_label)

        if log_scale:
            axes.set_yscale('log')

        axes.set_xticks(list(positions), labels, rotation=45, ha='right')
        axes.grid(axis='y', alpha=0.3)
        axes.legend()

        figure.tight_layout()

        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        figure.savefig(file_path)
        plt.close(figure)

        return file_path

    def results_table(self, rows: List[Row]) -> pd.DataFrame:
        """Every result as a table, one row per instance, in the same order the charts use.
        """
        return pd.DataFrame(self._ordered(rows))

    def _ordered(self, rows: List[Row]) -> List[Row]:
        return sorted(rows, key=lambda row: (row['n'], row['instance']))

    def _label(self, row: Row, format_instance: Optional[Callable[[str], str]]) -> str:
        name = row['instance']

        return name if format_instance is None else format_instance(name)

    def _values(self, rows: List[Row], column: str) -> List[float|int]:
        if not any(column in row for row in rows):
            raise ValueError(f"no result has a column named '{column}'")

        return [float('nan') if row.get(column) is None else int(row[column]) for row in rows]

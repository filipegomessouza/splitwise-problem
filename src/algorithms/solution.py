from dataclasses import dataclass
import os
import warnings
import graphviz
import numpy as np
from src.constants.role import Role
from src.instance.instance import Instance

WARN_ABOVE_PEOPLE = 200

LEGEND = 'P: person    B: balance'

@dataclass
class Solution:
    """The transactions that settle an instance. Counterpart of Instance.

    Transactions are held as three parallel arrays rather than a list of triples: it is
    what lets validate settle everything with two scatter-adds, and what lets a local
    search score a whole neighbourhood without unpacking anything. Amounts keep whatever
    dtype the algorithm produced -- integers from the greedy, floats from the solver.
    """
    instance: Instance
    payers: np.ndarray
    receivers: np.ndarray
    amounts: np.ndarray

    def __post_init__(self) -> None:
        self.payers = np.asarray(self.payers, dtype=np.int64)
        self.receivers = np.asarray(self.receivers, dtype=np.int64)
        self.amounts = np.asarray(self.amounts)

    @property
    def people(self) -> int:
        return len(self.instance.balances)

    @property
    def fitness(self) -> int:
        return len(self.payers)

    def validate(self) -> None:
        """Raise unless the transactions net out to exactly each person's balance.

        Checks the settlement, not its cost, so it holds for any algorithm's output.
        """
        if not (len(self.payers) == len(self.receivers) == len(self.amounts)):
            raise ValueError('payers, receivers and amounts must be the same length')

        unknown = (
            (self.payers < 0) | (self.payers >= self.people)
            | (self.receivers < 0) | (self.receivers >= self.people)
        )

        self._reject(unknown, 'refers to an unknown person')
        self._reject(self.payers == self.receivers, 'has the same payer and receiver')
        self._reject(self.amounts <= 0, 'must transfer a positive amount')

        net = np.zeros(self.people, dtype=self.amounts.dtype)

        # scatter-add rather than += so that repeated indices accumulate instead of
        # each one overwriting the last
        np.add.at(net, self.payers, -self.amounts)
        np.add.at(net, self.receivers, self.amounts)

        if not np.array_equal(net, self.instance.balances):
            person = int(np.argmax(net != self.instance.balances))
            raise ValueError(
                f"person {person} settled {net[person]} instead of "
                f"{self.instance.balances[person]}"
            )

    def _reject(self, offending: np.ndarray, complaint: str) -> None:
        if not offending.any():
            return

        # argmax on a boolean array finds the first True, so the message names the same
        # transaction a sequential check would have stopped at
        index = int(np.argmax(offending))

        raise ValueError(
            f"transaction ({self.payers[index]}, {self.receivers[index]}, "
            f"{self.amounts[index]}) {complaint}"
        )

    def describe(self) -> str:
        lines = [
            f"{self.people} people, {self.fitness} transactions",
            '',
            f"{'person':>7} {'balance':>9} {'role':>9}",
        ]

        for person, balance in enumerate(self.instance.balances):
            lines.append(f"{person:>7} {balance:>9} {Role.of(balance).value:>9}")

        lines.append('')
        lines.append('transactions:')

        for payer, receiver, amount in zip(self.payers, self.receivers, self.amounts):
            lines.append(f"  person {payer} -> person {receiver}: {amount}")

        return '\n'.join(lines)

    def render(self, file_path: str, format: str = 'png', engine: str = 'dot') -> str:
        """Write a directed graph of the transactions and return the path written.

        Nodes are people, edges are transfers labelled with their amount. The default
        ranked layout separates payers from receivers, which reads well on small
        instances but degenerates into a very wide, very short image as they grow --
        above WARN_ABOVE_PEOPLE people prefer format='svg', engine='sfdp'.
        """
        if self.people > WARN_ABOVE_PEOPLE:
            message = (
                f"rendering {self.people} people with engine='{engine}', format='{format}' "
                f"may take tens of seconds and produce a barely readable image"
            )

            if (engine, format) != ('sfdp', 'svg'):
                message += "; consider format='svg', engine='sfdp'"

            warnings.warn(message, stacklevel=2)

        graph = graphviz.Digraph(engine=engine, format=format)
        graph.attr(label=LEGEND, labelloc='t')
        graph.attr('node', shape='box', style='filled')

        for person, balance in enumerate(self.instance.balances):
            graph.node(
                str(person),
                label=f"P: {person}\nB: {balance}",
                fillcolor=Role.of(balance).color(),
            )

        for payer, receiver, amount in zip(self.payers, self.receivers, self.amounts):
            graph.edge(str(payer), str(receiver), label=str(amount))

        # graphviz appends the format as extension, so an already-suffixed path would
        # otherwise end up as graph.png.png
        stem = file_path
        suffix = f".{format}"

        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

        directory = os.path.dirname(stem)

        if directory:
            os.makedirs(directory, exist_ok=True)

        return graph.render(stem, cleanup=True)

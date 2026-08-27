from dataclasses import dataclass
import os
import warnings
import graphviz
from src.constants.role import Role
from src.constants.types import TransactionList
from src.instance.instance import Instance

WARN_ABOVE_PEOPLE = 200

LEGEND = 'P: person    C: contributed    B: balance'

@dataclass
class Solution:
    """The transactions that settle an instance. Counterpart of Instance."""
    instance: Instance
    transactions: TransactionList

    @property
    def people(self) -> int:
        return len(self.instance.contributions)

    @property
    def fitness(self) -> int:
        return len(self.transactions)

    def validate(self) -> None:
        """Raise unless the transactions leave everyone having disbursed exactly the mean.

        Checks the settlement, not its cost, so it holds for any algorithm's output.
        """
        net = [0] * self.people

        for payer, receiver, amount in self.transactions:
            if not 0 <= payer < self.people or not 0 <= receiver < self.people:
                raise ValueError(f"transaction ({payer}, {receiver}, {amount}) refers to an unknown person")

            if payer == receiver:
                raise ValueError(f"transaction ({payer}, {receiver}, {amount}) has the same payer and receiver")

            if amount <= 0:
                raise ValueError(f"transaction ({payer}, {receiver}, {amount}) must transfer a positive amount")

            net[payer] -= amount
            net[receiver] += amount

        mean = self.instance.mean

        for person, contribution in enumerate(self.instance.contributions):
            disbursed = contribution - net[person]

            if disbursed != mean:
                raise ValueError(f"person {person} disbursed {disbursed} instead of the mean {mean}")

    def describe(self) -> str:
        balances = self.instance.balances

        lines = [
            f"{self.people} people, mean {self.instance.mean}, {self.fitness} transactions",
            '',
            f"{'person':>7} {'contributed':>12} {'balance':>9} {'role':>9}",
        ]

        for person, contribution in enumerate(self.instance.contributions):
            balance = balances[person]
            lines.append(f"{person:>7} {contribution:>12} {balance:>9} {Role.of(balance).value:>9}")

        lines.append('')
        lines.append('transactions:')

        for payer, receiver, amount in self.transactions:
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

        balances = self.instance.balances

        for person, contribution in enumerate(self.instance.contributions):
            balance = balances[person]
            graph.node(
                str(person),
                label=f"P: {person}\nC: {contribution}\nB: {balance}",
                fillcolor=Role.of(balance).color(),
            )

        for payer, receiver, amount in self.transactions:
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

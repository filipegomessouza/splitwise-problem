from dataclasses import dataclass
import os
import warnings
import graphviz
from src.constants.role import Role
from src.constants.types import TransactionList
from src.instance.instance import Instance

WARN_ABOVE_PEOPLE = 200

LEGEND = 'P: person    B: balance'

@dataclass
class Solution:
    """The transactions that settle an instance. Counterpart of Instance."""
    instance: Instance
    transactions: TransactionList

    @property
    def people(self) -> int:
        return len(self.instance.balances)

    @property
    def fitness(self) -> int:
        return len(self.transactions)

    def validate(self) -> None:
        """Raise unless the transactions net out to exactly each person's balance.

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

        for person, balance in enumerate(self.instance.balances):
            if net[person] != balance:
                raise ValueError(f"person {person} settled {net[person]} instead of {balance}")

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

        for person, balance in enumerate(self.instance.balances):
            graph.node(
                str(person),
                label=f"P: {person}\nB: {balance}",
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

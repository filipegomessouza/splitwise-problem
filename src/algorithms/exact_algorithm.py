import gurobipy as gp
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.solution import Solution
from src.instance.instance import Instance
from src.constants.types import TransactionList

class ExactAlgorithm(BaseAlgorithm):
    def __init__(self, instance: Instance):
        self._instance = instance

    def run(self) -> Solution:

        model = gp.Model("splitwise")

        I = range(len(self._instance.contributions))
        J = range(len(self._instance.contributions))

        X = model.addVars(I, J, vtype=gp.GRB.CONTINUOUS, name="X")
        Y = model.addVars(I, J, vtype=gp.GRB.BINARY, name="Y")

        model.setObjective(
            gp.quicksum(Y[i, j] for i in I for j in J if i != j),
            sense=gp.GRB.MINIMIZE
        )

        balance = self._instance.balances

        model.addConstrs(
            (gp.quicksum(X[i, j] for j in J if i != j) - gp.quicksum(X[j, i] for j in J if i != j)
             == -balance[i] for i in I),
            name="balance_constraints"
        )

        model.addConstrs(
            (X[i, j] <= max(balance[j], 0) * Y[i, j] for i in I for j in J if i != j),
            name="link_constraints"
        )

        model.addConstrs(
            (X[i, j] >= 0 for i in I for j in J if i != j),
            name="nonnegativity_constraints"
        )

        model.optimize()

        transactions: TransactionList = [
            (i, j, X[i, j].X) for i in I for j in J if i != j and Y[i, j].X > 0.5
        ]

        return Solution(instance=self._instance, transactions=transactions)

from typing import Optional
import gurobipy as gp
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.instance.instance import Instance
from src.constants.types import TransactionList

# the size-limited license caps the model, and this model uses 2n^2 variables
LICENSE_VARIABLE_LIMIT = 2000

STATUS_NAMES = {
    gp.GRB.OPTIMAL: 'optimal',
    gp.GRB.INFEASIBLE: 'infeasible',
    gp.GRB.INF_OR_UNBD: 'infeasible_or_unbounded',
    gp.GRB.UNBOUNDED: 'unbounded',
    gp.GRB.TIME_LIMIT: 'time_limit',
    gp.GRB.INTERRUPTED: 'interrupted',
}

# the solver returns floats, so a transfer is only real if it is meaningfully above zero
TRANSFER_TOLERANCE = 1e-6

class ExactAlgorithm(BaseAlgorithm):
    def __init__(self, time_limit: Optional[float] = None, verbose: bool = False) -> None:
        self._time_limit = time_limit
        self._verbose = verbose

    def name(self) -> str:
        return 'exact'

    def supports(self, instance: Instance) -> bool:
        return 2 * len(instance.balances) ** 2 <= LICENSE_VARIABLE_LIMIT

    def run(self, instance: Instance) -> RunResult:
        # silenced from the environment up, otherwise the license banner escapes before
        # a model-level OutputFlag could take effect
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 1 if self._verbose else 0)
        env.start()

        model = gp.Model("splitwise", env=env)

        if self._time_limit is not None:
            model.Params.TimeLimit = self._time_limit

        I = range(len(instance.balances))
        J = range(len(instance.balances))

        X = model.addVars(I, J, vtype=gp.GRB.CONTINUOUS, name="X")
        Y = model.addVars(I, J, vtype=gp.GRB.BINARY, name="Y")

        model.setObjective(
            gp.quicksum(Y[i, j] for i in I for j in J if i != j),
            sense=gp.GRB.MINIMIZE
        )

        balance = instance.balances

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

        status = STATUS_NAMES.get(model.Status, f"status_{model.Status}")

        # reading .X without an incumbent raises an opaque AttributeError
        if model.SolCount == 0:
            raise RuntimeError(f"no feasible solution found (status: {status})")

        transactions: TransactionList = [
            (i, j, X[i, j].X)
            for i in I for j in J
            if i != j and Y[i, j].X > 0.5 and X[i, j].X > TRANSFER_TOLERANCE
        ]

        solution = Solution(instance=instance, transactions=transactions)

        return RunResult(solution=solution, status=status, gap=model.MIPGap)

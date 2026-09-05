from typing import Optional
import gurobipy as gp
import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.instance.instance import Instance

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

        balance = instance.balances
        people = len(balance)

        # a self-transfer settles nothing, so the diagonal is pinned shut through the
        # bounds -- which also lets the objective and the flow sums run over the whole
        # matrix, since those entries contribute zero
        transfer_limit = np.full((people, people), gp.GRB.INFINITY)
        indicator_limit = np.ones((people, people))
        np.fill_diagonal(transfer_limit, 0.0)
        np.fill_diagonal(indicator_limit, 0.0)

        X = model.addMVar((people, people), lb=0.0, ub=transfer_limit, name="X")
        Y = model.addMVar((people, people), ub=indicator_limit, vtype=gp.GRB.BINARY, name="Y")

        model.setObjective(Y.sum(), sense=gp.GRB.MINIMIZE)

        model.addConstr(X.sum(axis=1) - X.sum(axis=0) == -balance, name="balance_constraints")

        # broadcasting along the columns gives X[i, j] <= max(balance[j], 0) * Y[i, j]:
        # nobody can be sent more than they are owed, and only over an open indicator
        model.addConstr(X <= np.maximum(balance, 0)[np.newaxis, :] * Y, name="link_constraints")

        # non-negativity rides on lb=0 above, where gurobi puts it by default anyway

        model.optimize()

        status = STATUS_NAMES.get(model.Status, f"status_{model.Status}")

        # reading .X without an incumbent raises an opaque AttributeError
        if model.SolCount == 0:
            raise RuntimeError(f"no feasible solution found (status: {status})")

        transfers = X.X
        transferred = (Y.X > 0.5) & (transfers > TRANSFER_TOLERANCE)
        payers, receivers = np.nonzero(transferred)

        solution = Solution(
            instance=instance,
            payers=payers,
            receivers=receivers,
            amounts=transfers[transferred],
        )

        return RunResult(solution=solution, status=status, gap=model.MIPGap)

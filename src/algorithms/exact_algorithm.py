from typing import List, Optional
import gurobipy as gp
import numpy as np
from src.algorithms.base_algorithm import METRICS, BaseAlgorithm
from src.algorithms.greedy_algorithm import GreedyAlgorithm
from src.algorithms.run_result import RunResult
from src.algorithms.solution import Solution
from src.helpers.direct_transactions import pair_direct_transactions
from src.instance.instance import Instance

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
    """Mixed-integer model over payer -> receiver transfers only.

    The optimum is N - k*, k* being the most zero-sum parts the balances partition into,
    and each part settles in |part| - 1 transfers that all go from a payer to a receiver --
    so no transfer through an intermediary is ever needed, and the model is d x c rather
    than n x n. People owing and owed the very same amount are paired off beforehand: some
    optimum always settles them with each other, so the solver never sees them.
    """

    def __init__(self, time_limit: Optional[float] = None, verbose: bool = False) -> None:
        self._time_limit = time_limit
        self._verbose = verbose

    def name(self) -> str:
        return 'exact'

    def metrics(self) -> List[str]:
        # alone among the algorithms, this one can say whether its answer is the optimum
        return METRICS + ['proven']

    def run(self, instance: Instance) -> RunResult:
        balance = instance.balances

        paired_payers, paired_receivers, paired_amounts, left = pair_direct_transactions(balance)

        payers = left[balance[left] < 0]
        receivers = left[balance[left] > 0]

        # the pairing settled everyone, and it is optimal on its own
        if payers.size == 0:
            solution = Solution(
                instance=instance,
                payers=paired_payers,
                receivers=paired_receivers,
                amounts=paired_amounts,
            )

            return RunResult(solution=solution, proven=True)

        owed = -balance[payers]
        due = balance[receivers]

        # silenced from the environment up, otherwise the license banner escapes before
        # a model-level OutputFlag could take effect
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 1 if self._verbose else 0)
        env.start()

        model = gp.Model("splitwise", env=env)

        if self._time_limit is not None:
            model.Params.TimeLimit = self._time_limit

        # rows are payers and columns receivers; non-negativity rides on lb=0
        X = model.addMVar((payers.size, receivers.size), lb=0.0, name="X")
        Y = model.addMVar((payers.size, receivers.size), vtype=gp.GRB.BINARY, name="Y")

        model.setObjective(Y.sum(), sense=gp.GRB.MINIMIZE)

        model.addConstr(X.sum(axis=1) == owed, name="payer_constraints")
        model.addConstr(X.sum(axis=0) == due, name="receiver_constraints")

        # no transfer can exceed what the payer owes nor what the receiver is due. Taking
        # the smaller of the two, rather than the receiver's alone, is what makes the LP
        # relaxation open at least one transfer per person by itself
        model.addConstr(X <= np.minimum.outer(owed, due) * Y, name="link_constraints")

        self._start_from_greedy(X, Y, payers, receivers, left, balance)

        model.optimize()

        status = STATUS_NAMES.get(model.Status, f"status_{model.Status}")

        # reading .X without an incumbent raises an opaque AttributeError
        if model.SolCount == 0:
            raise RuntimeError(f"no feasible solution found (status: {status})")

        transfers = X.X
        transferred = (Y.X > 0.5) & (transfers > TRANSFER_TOLERANCE)
        rows, columns = np.nonzero(transferred)

        solution = Solution(
            instance=instance,
            payers=np.concatenate((paired_payers, payers[rows])),
            receivers=np.concatenate((paired_receivers, receivers[columns])),
            amounts=np.concatenate((paired_amounts, transfers[transferred])),
        )

        # anything short of OPTIMAL means the objective is an incumbent: an upper bound
        # that may or may not be the optimum, with no way to tell from here
        return RunResult(solution=solution, proven=model.Status == gp.GRB.OPTIMAL)

    def _start_from_greedy(
        self,
        X: gp.MVar,
        Y: gp.MVar,
        payers: np.ndarray,
        receivers: np.ndarray,
        left: np.ndarray,
        balance: np.ndarray,
    ) -> None:
        """Hand the solver the greedy's settlement as its first incumbent.

        Under a time limit this keeps the answer from ever being worse than the greedy's,
        and gives the search a bound to prune against from the very first node. Every
        greedy step settles at least one of its two people for good, so no pair repeats
        and each transfer fits a single cell.
        """
        greedy_payers, greedy_receivers, greedy_amounts = GreedyAlgorithm().settle_largest_first(
            left, balance[left]
        )

        row_of = np.empty(len(balance), dtype=np.int64)
        column_of = np.empty(len(balance), dtype=np.int64)
        row_of[payers] = np.arange(payers.size)
        column_of[receivers] = np.arange(receivers.size)

        transfer_start = np.zeros(X.shape)
        indicator_start = np.zeros(Y.shape)
        transfer_start[row_of[greedy_payers], column_of[greedy_receivers]] = greedy_amounts
        indicator_start[row_of[greedy_payers], column_of[greedy_receivers]] = 1.0

        X.Start = transfer_start
        Y.Start = indicator_start

from typing import List, Tuple

Transaction = Tuple[int, int, int|float]  # (payer, receiver, amount)
TransactionList = List[Transaction]

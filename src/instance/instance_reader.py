import numpy as np
from src.instance.instance import Instance

class InstanceReader:
    def read(self, file_path: str) -> Instance:
        balances = np.loadtxt(file_path, dtype=np.int64, ndmin=1)

        return Instance(balances)

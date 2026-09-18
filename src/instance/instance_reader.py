import numpy as np
from src.instance.instance import Instance

class InstanceReader:
    def read(self, file_path: str) -> Instance:
        balances = np.loadtxt(file_path, dtype=np.int64, ndmin=1)

        return Instance(balances, name=self._instance_name(file_path))

    def read_with_index(self, file_path: str) -> Instance:
        lines = np.loadtxt(file_path, dtype=np.int64, ndmin=2)
        balances = lines[:, 1]

        return Instance(balances, name=self._instance_name(file_path))

    def _instance_name(self, file_path: str) -> str:
        return file_path.split('/')[-1].split('.')[0]

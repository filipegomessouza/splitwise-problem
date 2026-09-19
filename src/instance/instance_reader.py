from io import StringIO
import numpy as np
from src.instance.instance import Instance

class InstanceReader:
    def read(self, file_path: str) -> Instance:
        balances = self._load(file_path, ndmin=1)

        return Instance(balances, name=self._instance_name(file_path))

    def read_with_index(self, file_path: str) -> Instance:
        lines = self._load(file_path, ndmin=2)
        balances = lines[:, 1]

        return Instance(balances, name=self._instance_name(file_path))

    def _load(self, file_path: str, ndmin: int) -> np.ndarray:
        with open(file_path) as file:
            columns = file.read().replace(',', ' ')

        return np.loadtxt(StringIO(columns), dtype=np.int64, ndmin=ndmin)

    def _instance_name(self, file_path: str) -> str:
        return file_path.split('/')[-1].split('.')[0]

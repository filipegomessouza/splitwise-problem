from src.instance.instance import Instance

class InstanceReader:
    def read(self, file_path: str) -> Instance:
        with open(file_path, 'r') as file:
            balances = [int(line) for line in file if line.strip()]

        return Instance(balances)

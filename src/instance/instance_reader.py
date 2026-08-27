from src.instance.instance import Instance

class InstanceReader:
    def read(self, file_path: str) -> Instance:
        with open(file_path, 'r') as file:
            values = [int(line.strip()) for line in file]

        return Instance(values)
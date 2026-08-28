import os
from src.instance.instance_factory import InstanceFactory

class InstanceSeeder:
    def seed(self) -> None:
        SEED = 42
        MAX_VALUE = 1000
        SIZES = [10, 20, 30, 40, 50, 100, 1000]

        for size in SIZES:
            instance_factory = InstanceFactory(max_value=MAX_VALUE, size=size, seed=SEED + size)
            instance_factory.create_as_txt(os.path.join('instances', f'{size}.txt'))

from src.instance.instance import Instance
from src.instance.instance_factory import InstanceFactory

class InstanceSeeder:
    def seed(self) -> None:
        SEED = 42
        MAX_VALUE = 1000
        SIZES = [10, 100, 1000, 10000]

        for size in SIZES:
            instance_factory = InstanceFactory(max_value=MAX_VALUE, size=size, seed=SEED)
            instance_factory.create_as_txt(f'instances/{size}.txt')

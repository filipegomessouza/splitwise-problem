import os
from src.instance.instance_factory import InstanceFactory

class InstanceSeeder:
    def seed(self) -> None:
        SEED = 42

        GRID = [
            (10, [1, 2, 5]),
            (20, [1, 2, 4, 5, 10]),
            (30, [1, 3, 5, 10, 15]),
            (50, [1, 5, 10, 25]),
            (100, [1, 5, 10, 25, 50]),
            (200, [1, 10, 20, 50, 100]),
            (300, [1, 15, 30, 75, 150]),
            (400, [1, 10, 20, 50, 200]),
            (500, [1, 25, 50, 125, 250]),
        ]

        i = 0

        for N, components in GRID:
            # accidental zero-sum subsets grow with 2^N, so the range has to grow too or
            # the instances with few components stop having few zero-sum subsets
            B = max(1000, 100 * N)

            for K in components:
                instance_factory = InstanceFactory(N=N, B=B, K=K, seed=SEED + i)

                i += 1

                file_name = f'n{N:04d}_b{B}_k{K:04d}.txt'
                instance_factory.create_as_txt(os.path.join('instances', file_name))

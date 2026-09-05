import os
from src.instance.instance_factory import InstanceFactory

class InstanceSeeder:
    def seed(self) -> None:
        SEED = 42
        PEOPLE = [10, 20, 30, 50, 100, 1000]
        STRUCTURES = [0.0, 0.25, 0.5, 0.75, 1.0]

        i = 0

        for people in PEOPLE:
            # accidental zero-sum subsets grow with 2^people, so the range has to grow
            # too or the low-structure instances stop being low-structure
            max_balance = max(1000, 100 * people)

            for structure in STRUCTURES:
                instance_factory = InstanceFactory(
                    people=people,
                    max_balance=max_balance,
                    structure=structure,
                    seed=SEED + i,
                )

                i += 1

                # people is zero-padded so that sorting the filenames sorts by size
                file_name = f'n{people:04d}_b{max_balance}_w{structure:.2f}.txt'
                instance_factory.create_as_txt(os.path.join('instances', file_name))

from src.algorithms.best_improvement_algorithm import BestImprovementAlgorithm
from src.algorithms.exact_algorithm import ExactAlgorithm
from src.algorithms.first_improvement_algorithm import FirstImprovementAlgorithm
from src.algorithms.greedy_algorithm import GreedyAlgorithm
from src.algorithms.random_key_algorithm import RandomKeyAlgorithm
from src.instance.instance_reader import InstanceReader
from src.runner.runner import Runner

SEED = 42
MAX_WORKERS = 4

INSTANCE_PATHS = [
    'instances_ufes/instancia_splitwise_20.txt',
    'instances_ufes/instancia_splitwise_30.txt',
    'instances_ufes/instancia_splitwise_30_5.txt',
    'instances_ufes/instancia_splitwise_50.txt',
    'instances_ufes/instancia_splitwise_50_8.txt',
    'instances_ufes/instancia_splitwise_100.txt',
    'instances_ufes/instancia_splitwise_100_20.txt',
    'instances_ufes/instancia_splitwise_150.txt',
    'instances_ufes/instancia_splitwise_150_15.txt',
    'instances_ufes/instancia_splitwise_200.txt',
    'instances_ufes/instancia_splitwise_200_8.txt',
    'instances_ufes/instancia_splitwise_200_80.txt',
    'instances_ufes/instancia_splitwise_300.txt',
    'instances_ufes/instancia_splitwise_300_60.txt',
    'instances_ufes/instancia_splitwise_400.txt',
    'instances_ufes/instancia_splitwise_400_35.txt',
    'instances_ufes/instancia_splitwise_500.txt',
    'instances_ufes/instancia_splitwise_500_25.txt',
    'instances_ufes/instancia_splitwise_500_188.txt',
    'instances_ufes/instancia_splitwise_600.txt',
    'instances_ufes/instancia_splitwise_600_40.txt',
    'instances_ufes/instancia_splitwise_700.txt',
    'instances_ufes/instancia_splitwise_700_90.txt',
    'instances_ufes/instancia_splitwise_800.txt',
    'instances_ufes/instancia_splitwise_800_80.txt',
    'instances_ufes/instancia_splitwise_900.txt',
    'instances_ufes/instancia_splitwise_900_100.txt',
    'instances_ufes/instancia_splitwise_1000.txt',
    'instances_ufes/instancia_splitwise_1000_120.txt',
    'instances_ufes/instancia_splitwise_1100.txt',
    'instances_ufes/instancia_splitwise_1300.txt',
    'instances_ufes/instancia_splitwise_1500.txt',
]

if __name__ == '__main__':
    instance_reader = InstanceReader()
    instances = [instance_reader.read_with_index(path) for path in INSTANCE_PATHS]

    algorithms = [
        GreedyAlgorithm(),
        RandomKeyAlgorithm(seed=SEED),
        BestImprovementAlgorithm(RandomKeyAlgorithm(seed=SEED)),
        FirstImprovementAlgorithm(RandomKeyAlgorithm(seed=SEED)),
        BestImprovementAlgorithm(GreedyAlgorithm(seed=SEED)),
        FirstImprovementAlgorithm(GreedyAlgorithm(seed=SEED)),
        # ExactAlgorithm(time_limit=30.0),
    ]

    Runner().run(
        instances,
        algorithms,
        output_path='results/results.json',
        max_workers=MAX_WORKERS,
    )

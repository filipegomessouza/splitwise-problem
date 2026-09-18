import glob
from src.algorithms.best_improvement_algorithm import BestImprovementAlgorithm
from src.algorithms.exact_algorithm import ExactAlgorithm
from src.algorithms.first_improvement_algorithm import FirstImprovementAlgorithm
from src.algorithms.greedy_algorithm import GreedyAlgorithm
from src.algorithms.random_key_algorithm import RandomKeyAlgorithm
from src.instance.instance_reader import InstanceReader
from src.runner.runner import Runner

SEED = 42
INSTANCE_PATHS = sorted(glob.glob('instances_ufes_sample/*.txt'), key=lambda path: int(path.split('/')[-1].split('_')[2].split('.')[0]))

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

results = Runner(instances, algorithms).run()

with open('results/results.txt', 'w') as file:
    file.write(results.to_string(index=False))

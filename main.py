import glob
from src.algorithms.exact_algorithm import ExactAlgorithm
from src.algorithms.greedy_algorithm import GreedyAlgorithm
from src.instance.instance_reader import InstanceReader
from src.runner.runner import Runner

# the seeder zero-pads the people count, so sorting the paths sorts by instance size
INSTANCE_PATHS = sorted(glob.glob('instances/*.txt'))

instance_reader = InstanceReader()

instances = [instance_reader.read(path) for path in INSTANCE_PATHS]
algorithms = [GreedyAlgorithm(), ExactAlgorithm(time_limit=30.0)]

results = Runner(instances, algorithms).run()

print(results.to_string(index=False))

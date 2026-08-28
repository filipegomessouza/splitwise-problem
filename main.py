from src.algorithms.exact_algorithm import ExactAlgorithm
from src.algorithms.greedy_algorithm import GreedyAlgorithm
from src.instance.instance_reader import InstanceReader
from src.runner.runner import Runner

INSTANCE_PATHS = [
    "instances/10.txt",
    "instances/20.txt",
    "instances/30.txt",
    "instances/40.txt",
    "instances/50.txt",
    "instances/100.txt",
    "instances/1000.txt",
]

instance_reader = InstanceReader()

instances = [instance_reader.read(path) for path in INSTANCE_PATHS]
algorithms = [GreedyAlgorithm(), ExactAlgorithm(time_limit=30.0)]

results = Runner(instances, algorithms).run()

print(results.to_string(index=False))

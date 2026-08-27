from src.instance.instance_reader import InstanceReader
from src.algorithms.greedy_algorithm import GreedyAlgorithm

instance_reader = InstanceReader()

instance = instance_reader.read("instances/10.txt")

greedy_algorithm = GreedyAlgorithm(instance)
solution = greedy_algorithm.run()
solution.validate()

print(solution.describe())
print()
print(f"graph written to {solution.render('output/solution')}")
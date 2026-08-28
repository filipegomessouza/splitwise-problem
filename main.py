from src.instance.instance_reader import InstanceReader
from src.algorithms.greedy_algorithm import GreedyAlgorithm
from src.algorithms.exact_algorithm import ExactAlgorithm

instance_reader = InstanceReader()

instance = instance_reader.read("instances/10.txt")

greedy_algorithm = GreedyAlgorithm(instance)
solution = greedy_algorithm.run()
solution.validate()

print(solution.describe())
print()
print(f"graph written to {solution.render('output/solution')}")

exact_algorithm = ExactAlgorithm(instance)
solution = exact_algorithm.run()
solution.validate()

print(solution.describe())
print()
print(f"graph written to {solution.render('output/solution_exact')}")

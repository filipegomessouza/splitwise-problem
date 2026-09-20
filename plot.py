import json
from src.analysis.analysis import Analysis

RESULTS_PATH = 'results/all_results.json'
INSTANCE_PREFIX = 'instancia_splitwise_'

GREEDY_FITNESS = {
    'greedy_fitness': 'Greedy',
    'greedy_bi_fitness': 'Best Improvement',
    'greedy_fi_fitness': 'First Improvement',
}
RANDOM_KEY_FITNESS = {
    'random_key_fitness': 'Random Keys',
    'random_key_bi_fitness': 'Best Improvement',
    'random_key_fi_fitness': 'First Improvement',
}

def short_name(instance: str) -> str:
    return instance.removeprefix(INSTANCE_PREFIX)

with open(RESULTS_PATH) as file:
    rows = json.load(file)

analysis = Analysis()

print(analysis.plot_results(
    rows,
    GREEDY_FITNESS,
    title='Results - Greedy Construction',
    file_path='output/fitness_greedy.eps',
    y_label='Transactions',
    format_instance=short_name,
))

print(analysis.plot_results(
    rows,
    RANDOM_KEY_FITNESS,
    title='Results - Random Key Construction',
    file_path='output/fitness_random_key.eps',
    y_label='Transactions',
    format_instance=short_name,
))

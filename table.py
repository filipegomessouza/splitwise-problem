import json
import pandas as pd
from src.analysis.analysis import Analysis

RESULTS_PATH = 'results/all_results.json'

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

with open(RESULTS_PATH) as file:
    rows = json.load(file)

table = Analysis().results_table(rows)

print(table.to_excel('results/results.xlsx', index=False))

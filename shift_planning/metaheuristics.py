import csv
import sys
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd


class GeneticAlgorithm:
    """
    Genetic Algorithm for Shift Planning
    """

    def __init__(
        self,
        input_dir: str = "./input_data/",
        problem_instance: int = 1,
    ):
        self._forced_day_off = pd.read_csv(f"{input_dir}/plan_{problem_instance}/forced_day_off.csv")
        self._pref_day_off = pd.read_csv(f"{input_dir}/plan_{problem_instance}/pref_day_off.csv")
        self._pref_work_shift = pd.read_csv(f"{input_dir}/plan_{problem_instance}/pref_work_shift.csv")
        self._qualified_route = pd.read_csv(f"{input_dir}/plan_{problem_instance}/qualified_route.csv")

    def run(self):
        return None

    def _move(self):
        return None

    def _mutation(self):
        return None

    def _crossover(self):
        return None
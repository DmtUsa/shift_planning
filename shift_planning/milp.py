import csv
import sys
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import pulp


class MILP:
    """
    MILP for Shift Planning
    """

    def __init__(
        self, input_dir: str = "./input_data/", problem_instance: int = 1,
    ):
        self._forced_day_off = pd.read_csv(
            f"{input_dir}/plan_{problem_instance}/forced_day_off.csv"
        )
        self._pref_day_off = pd.read_csv(
            f"{input_dir}/plan_{problem_instance}/pref_day_off.csv"
        )
        self._pref_work_shift = pd.read_csv(
            f"{input_dir}/plan_{problem_instance}/pref_work_shift.csv"
        )
        self._qualified_route = pd.read_csv(
            f"{input_dir}/plan_{problem_instance}/qualified_route.csv"
        )

    def solve(self):
        nr_of_couriers = self._forced_day_off.shape[0]
        nr_of_days = self._forced_day_off.shape[1]
        nr_of_routes = self._qualified_route.shape[1]

        # set of couriers
        C = range(nr_of_couriers)
        # set of routes
        R = range(nr_of_routes)
        # set of days
        D = range(nr_of_days)
        # set of shifts
        S = [1, 2]

        # qualification
        q = {(i, j): self._qualified_route.iloc[i, j + 1] for i in C for j in R}

        # vacation / forced days off
        v = {(i, k): self._forced_day_off.iloc[i, k + 1] for i in C for k in D}

        # preferred shifts
        s = {
            (i, k, l): int(self._pref_work_shift.iloc[i, k + 1] == l)
            for i in C
            for k in D
            for l in S
        }

        # preferred days of
        d = {(i, k): self._pref_day_off.iloc[i, k+1] for i in C for k in D}

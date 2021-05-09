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

        opt_model = pulp.LpProblem(name="pizza model")

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
        d = {(i, k): self._pref_day_off.iloc[i, k + 1] for i in C for k in D}

        # decision variables x_i_j_k_l
        # indicating if a courier i is assigned to a shift l on route j on day k
        x = {
            (i, j, k, l): pulp.LpVariable(cat=pulp.LpBinary, name=f"x_{i}_{j}_{k}_{l}")
            for i in C
            for j in R
            for k in D
            for l in S
        }

        # decision variables y_i_k
        # indicating if a courier i worked a night shift on day k followed by a daytime shift
        y = {
            (i, k): pulp.LpVariable(cat=pulp.LpBinary, name=f"y_{i}_{k}")
            for i in C
            for k in D
        }

        # decision variables z_i_k
        # indicating if a courier i is assigned to three consecutive night shifts starting on day k
        z = {
            (i, k): pulp.LpVariable(cat=pulp.LpBinary, name=f"z_{i}_{k}")
            for i in C
            for k in D
        }

        # Each route needs two couriers every day
        constraints_1 = {
            {j, k, l}: opt_model.addConstraint(
                pulp.LpConstraint(
                    e=pulp.lpSum(x[i, j, k, l] for i in C),
                    sense=pulp.LpConstraintEQ,
                    rhs=1,
                    name=f"1_constraint_{j}_{k}_{l}",
                )
            )
            for j in R
            for k in D
            for l in S
        }

        # Only one shift per day
        constraints_2 = {
            {i, k}: opt_model.addConstraint(
                pulp.LpConstraint(
                    e=pulp.lpSum(x[i, j, k, l] for j in R for l in S),
                    sense=pulp.LpConstraintLE,
                    rhs=1,
                    name=f"2_constraint_{i}_{k}",
                )
            )
            for i in R
            for k in D
        }

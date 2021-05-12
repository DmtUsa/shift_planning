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

    def run(self):

        nr_of_couriers = self._forced_day_off.shape[0]
        nr_of_days = self._forced_day_off.shape[1] - 1
        nr_of_routes = self._qualified_route.shape[1] - 1

        # generate initial solution
        solution = np.zeros((nr_of_couriers, nr_of_routes, nr_of_days, 2))
        for j in range(nr_of_routes):
            for k in range(nr_of_days):
                for l in range(2):
                    feasible_couriers = self._feasible_couriers(j, k, l, solution)
                    # pick a courier who is qualified for the least number of routes
                    qualification = self._qualified_route.iloc[
                        feasible_couriers, 1:
                    ].sum(axis=1)
                    least_qualified = qualification.loc[
                        qualification == qualification.min()
                    ].index.values
                    # out of those least qualified pick the one with least free days left
                    days_off = self._forced_day_off.iloc[
                        least_qualified, (k + 1):
                    ].sum(axis=1)
                    least_free = days_off.loc[days_off == days_off.min()].index.values[
                        0
                    ]
                    # assign one of the feasible couriers if any
                    if least_free.size > 0:
                        solution[least_free, j, k, l] = 1
                    # otherwise a random one
                    else:
                        solution[np.random.choice(np.arange(11)), j, k, l] = 1
        return None

    def _feasible_couriers(self, j, k, l, solution):
        qualified = self._qualified_route.index[
            self._qualified_route.iloc[:, j + 1] == 1
        ].values
        free = self._forced_day_off.index[
            self._forced_day_off.iloc[:, k + 1] == 0
        ].values
        not_scheduled_yet = np.where(solution[:, :, k, :].sum(axis=(1, 2)) == 0)[0]
        if False: #l == 1:
            allowed_for_the_shift = np.where(solution[:, :, :, 1].sum(axis=(1, 2)) < 4)[
                0
            ]
        else:
            allowed_for_the_shift = np.arange(11)

        return np.intersect1d(
            np.intersect1d(np.intersect1d(qualified, free), not_scheduled_yet),
            allowed_for_the_shift,
        )

    def _fitness(self):
        return None

    def _move(self):
        return None

    def _mutation(self):
        return None

    def _crossover(self):
        return None

    def _subarray_count(self, A, B):
        """
        Number of time array B is a subarray of A
        """

        n = A.size
        m = B.size
        counter = 0

        # Two pointers to traverse the arrays
        i = 0;
        j = 0;

        # Traverse both arrays simultaneously
        while i < n and j < m:

            # If element matches increment both pointers
            if A[i] == B[j]:

                i += 1
                j += 1

                # If array B is completely traversed
                if j == m:
                    counter += 1
                    # increment i and reset j
                    i = i - j + 1
                    j = 0
                    # return True

            # If not, increment i and reset j
            else:
                i = i - j + 1
                j = 0

        return counter

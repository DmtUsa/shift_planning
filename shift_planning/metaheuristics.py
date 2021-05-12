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
        ).iloc[:, 1:].values
        self._pref_day_off = pd.read_csv(
            f"{input_dir}/plan_{problem_instance}/pref_day_off.csv"
        ).iloc[:, 1:].values
        self._pref_work_shift = pd.read_csv(
            f"{input_dir}/plan_{problem_instance}/pref_work_shift.csv"
        ).iloc[:, 1:].values
        self._qualified_route = pd.read_csv(
            f"{input_dir}/plan_{problem_instance}/qualified_route.csv"
        ).iloc[:, 1:].values

    def run(self):

        nr_of_couriers = self._forced_day_off.shape[0]
        nr_of_days = self._forced_day_off.shape[1]
        nr_of_routes = self._qualified_route.shape[1]

        # generate initial solution
        solution = np.zeros((nr_of_couriers, nr_of_routes, nr_of_days, 2))
        for j in range(nr_of_routes):
            for k in range(nr_of_days):
                for l in range(2):
                    feasible_couriers = self._feasible_couriers(j, k, l, solution)
                    # pick a courier who is qualified for the least number of routes
                    qualification = self._qualified_route[
                        feasible_couriers, :
                    ].sum(axis=1)
                    least_qualified = feasible_couriers[
                        qualification == qualification.min()
                    ]
                    # out of those least qualified pick the one with least free days left
                    days_off = self._forced_day_off[
                        least_qualified, k:
                    ].sum(axis=1)
                    least_free = least_qualified[days_off == days_off.min()][0]
                    # assign one of the feasible couriers if any
                    if least_free.size > 0:
                        solution[least_free, j, k, l] = 1
                    # otherwise a random one
                    else:
                        solution[np.random.choice(np.arange(11)), j, k, l] = 1
        return None

    def _feasible_couriers(self, j, k, l, solution, ignore_night_shifts: bool = True):
        qualified = np.where(self._qualified_route[:, j] == 1)[0]
        free = np.where(self._forced_day_off[:, k] == 0)[0]
        not_scheduled_yet = np.where(solution[:, :, k, :].sum(axis=(1, 2)) == 0)[0]
        if not ignore_night_shifts and l == 1:
            allowed_for_the_shift = np.where(solution[:, :, :, 1].sum(axis=(1, 2)) < 4)[
                0
            ]
        else:
            allowed_for_the_shift = np.arange(11)

        return np.intersect1d(
            np.intersect1d(np.intersect1d(qualified, free), not_scheduled_yet),
            allowed_for_the_shift,
        )

    def _fitness(self, solution):
        fitness = 0
        # days off preference
        # self._pref_day_off[]
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

import copy
import csv
import itertools
import sys
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class GeneticAlgorithm:
    """
    Genetic Algorithm for Shift Planning
    """

    def __init__(
        self, input_dir: str = "./input_data/", problem_instance: int = 1,
    ):
        self._forced_day_off = (
            pd.read_csv(f"{input_dir}/plan_{problem_instance}/forced_day_off.csv")
            .iloc[:, 1:]
            .values
        )
        self._pref_day_off = (
            pd.read_csv(f"{input_dir}/plan_{problem_instance}/pref_day_off.csv")
            .iloc[:, 1:]
            .values
        )
        pref_work_shift = (
            pd.read_csv(f"{input_dir}/plan_{problem_instance}/pref_work_shift.csv")
            .iloc[:, 1:]
            .values
        )
        self._pref_work_shift = np.zeros(
            (self._forced_day_off.shape[0], self._forced_day_off.shape[1], 2)
        )
        self._pref_work_shift[:, :, 0] = (pref_work_shift == 1).astype(int)
        self._pref_work_shift[:, :, 1] = (pref_work_shift == 2).astype(int)
        self._qualified_route = (
            pd.read_csv(f"{input_dir}/plan_{problem_instance}/qualified_route.csv")
            .iloc[:, 1:]
            .values
        )

    def run(self, population_size: int = 100, best_candidates_fraction: float = 0.8):

        nr_of_couriers = self._forced_day_off.shape[0]
        nr_of_days = self._forced_day_off.shape[1]
        nr_of_routes = self._qualified_route.shape[1]

        best_candidates_number = round(best_candidates_fraction * population_size)

        # generate initial solution
        solution = np.zeros((nr_of_couriers, nr_of_routes, nr_of_days, 2))
        for j in range(nr_of_routes):
            for k in range(nr_of_days):
                for l in range(2):
                    feasible_couriers = self._feasible_couriers(j, k, l, solution)
                    # pick a courier who is qualified for the least number of routes
                    qualification = self._qualified_route[feasible_couriers, :].sum(
                        axis=1
                    )
                    least_qualified = feasible_couriers[
                        qualification == qualification.min()
                    ]
                    # out of those least qualified pick the one with least free days left
                    days_off = self._forced_day_off[least_qualified, k:].sum(axis=1)
                    least_free = least_qualified[days_off == days_off.min()][0]
                    # assign one of the feasible couriers if any
                    if least_free.size > 0:
                        solution[least_free, j, k, l] = 1
                    # otherwise a random one
                    else:
                        solution[np.random.choice(np.arange(11)), j, k, l] = 1
        initial_fitness = self._fitness(solution)

        # initial population
        population = [solution for x in range(population_size)]
        population_fitness = [initial_fitness for x in range(population_size)]

        best_fitness_history = [initial_fitness]

        while True:
            # apply move operator
            move_population = [self._move(copy.deepcopy(x)) for x in population]
            # apply crossover operator
            crossover_population = [
                self._crossover(copy.deepcopy(move_population))
                for x in range(int(len(move_population) / 2))
            ]
            crossover_population = list(itertools.chain(*crossover_population))
            new_population_candidates = (
                population + move_population + crossover_population
            )
            candidates_fitness = [self._fitness(x) for x in new_population_candidates]

            best_candidates_ids = sorted(
                range(len(candidates_fitness)), key=lambda i: candidates_fitness[i]
            )[-best_candidates_number:]
            random_candidates_ids = list(np.random.choice(
                [
                    x
                    for x in range(len(candidates_fitness))
                    if x not in best_candidates_ids
                ],
                population_size - best_candidates_number,
            ))
            population = [new_population_candidates[x] for x in best_candidates_ids+random_candidates_ids]
            best_fitness_history += max(candidates_fitness)

            plt.plot(best_fitness_history)
            pass
        return None

    def _fitness(self, solution):
        fitness = 0
        # benefits for days off preferences
        fitness += 4 * (self._pref_day_off * solution.sum(axis=(1, 3))).sum()
        # benefits for shift preferences
        fitness += 3 * (self._pref_work_shift * solution.sum(axis=1)).sum()
        # penalties for day shift following a night shift
        fitness -= 20 * (solution[:, :, :-1, 1] + solution[:, :, 1:, 0] > 1).sum()
        # penalties for each consecutive night shift after 3 consecutive night shifts
        fitness -= (
            10
            * np.apply_along_axis(
                self._subarray_count,
                axis=1,
                arr=solution[:, :, :, 1].sum(axis=1),
                B=np.array([1, 1, 1, 1]),
            ).sum()
        )
        # penalties for having more than 4 night shifts over the two week period
        fitness -= 100 * (solution[:, :, :, 1].sum(axis=(1, 2)) > 4).sum()

        return fitness

    def _move(self, solution):
        # pick a random shift
        (j, k, l) = (
            np.random.choice(solution.shape[1]),
            np.random.choice(solution.shape[2]),
            np.random.choice(2),
        )
        # currently assigned courier to that shift
        current_courier = np.where(solution[:, j, k, l] == 1)[0][0]

        # get the unassigned feasible couriers
        unassigned_feasible_couriers = self._feasible_couriers(
            j, k, l, solution, ignore_night_shifts=False
        )

        # get the assigned couriers feasible for exchange
        assigned_feasible_couriers = self._feasible_couriers(
            j, k, l, solution, ignore_night_shifts=False, consider_only_unassigned=False
        )
        assigned_feasible_couriers = assigned_feasible_couriers[
            assigned_feasible_couriers != current_courier
        ]
        # check which of those can the current courier be exchanged with
        assigned_shifts = np.where(solution[assigned_feasible_couriers, :, k, :] > 0)[
            1:
        ]
        temp_solution = copy.deepcopy(solution)
        temp_solution[current_courier, j, k, l] = 0
        current_courier_is_exchangable_with = np.array(
            [
                assigned_feasible_couriers[x]
                for x in range(assigned_feasible_couriers.size)
                if np.isin(
                    current_courier,
                    self._feasible_couriers(
                        assigned_shifts[0][x],
                        k,
                        assigned_shifts[1][x],
                        temp_solution,
                        ignore_night_shifts=False,
                    ),
                )
            ]
        )

        # pick a random (either unassigned or assigned) courier
        nr_of_alternatives = (
            unassigned_feasible_couriers.size + current_courier_is_exchangable_with.size
        )
        if nr_of_alternatives > 0:
            choice = np.random.choice(nr_of_alternatives)
            if choice < unassigned_feasible_couriers.size:
                alternative = unassigned_feasible_couriers[choice]
                solution[current_courier, j, k, l] = 0
                solution[alternative, j, k, l] = 1
            else:
                alternative = current_courier_is_exchangable_with[
                    choice - unassigned_feasible_couriers.size
                ]
                alternative_shift = np.where(solution[alternative, :, k, :] > 0)
                # reassign the current courier to the new (alternative) shift
                solution[current_courier, j, k, l] = 0
                solution[
                    current_courier, alternative_shift[0][0], k, alternative_shift[1][0]
                ] = 1
                # assign the alternative courier to the current shift
                solution[
                    alternative, alternative_shift[0][0], k, alternative_shift[1][0]
                ] = 0
                solution[alternative, j, k, l] = 1

        return solution

    def _mutation(self):
        return None

    def _crossover(self, population):
        # select two different solutions at random
        parents_indices = np.random.choice(len(population), 2, replace=False)
        parent_1 = population[parents_indices[0]]
        parent_2 = population[parents_indices[1]]

        # select a cutoff day at random
        number_of_days = population[0].shape[2]
        cutoff_day = np.random.choice(np.arange(1, number_of_days - 1))

        # child solutions
        child_1 = copy.deepcopy(parent_1)
        child_1[:, :, :cutoff_day, :] = parent_2[:, :, :cutoff_day, :]
        child_2 = copy.deepcopy(parent_1)
        child_2[:, :, cutoff_day:, :] = parent_2[:, :, cutoff_day:, :]

        return [child_1, child_2]

    def _feasible_couriers(
        self,
        j,
        k,
        l,
        solution,
        ignore_night_shifts: bool = True,
        consider_only_unassigned: bool = True,
    ):
        qualified = np.where(self._qualified_route[:, j] == 1)[0]
        free = np.where(self._forced_day_off[:, k] == 0)[0]
        if not ignore_night_shifts and l == 1:
            allowed_for_the_shift = np.where(solution[:, :, :, 1].sum(axis=(1, 2)) < 4)[
                0
            ]
        else:
            allowed_for_the_shift = np.arange(11)
        if consider_only_unassigned:
            not_scheduled_yet = np.where(solution[:, :, k, :].sum(axis=(1, 2)) == 0)[0]
            feasible_couriers = np.intersect1d(
                np.intersect1d(np.intersect1d(qualified, free), not_scheduled_yet),
                allowed_for_the_shift,
            )
        else:
            scheduled = np.where(solution[:, :, k, :].sum(axis=(1, 2)) == 1)[0]
            feasible_couriers = np.intersect1d(
                np.intersect1d(np.intersect1d(qualified, free), scheduled),
                allowed_for_the_shift,
            )

        return feasible_couriers

    def _subarray_count(self, A, B):
        """
        Number of time array B is a subarray of A
        """

        n = A.size
        m = B.size
        counter = 0

        # Two pointers to traverse the arrays
        i = 0
        j = 0

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

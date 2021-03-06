import copy
import itertools
from typing import Dict, List

import numpy as np
import pandas as pd


class GeneticAlgorithm:
    """
    Genetic Algorithm (GA) for Shift Planning
    """

    def __init__(
        self,
        population_size: int = 150,
        best_candidates_fraction: float = 0.5,
        stop_after: float = 15,
        mutation_threshold: float = 0.5,
        input_dir: str = "./input_data/",
        problem_instance: int = 1,
    ):
        """
        :param population_size: the number of solutions kept in a population for the next iteration of GA
        :param best_candidates_fraction: the fraction of a population that should contain
                                         the best solutions found in the previous iteration
        :param stop_after: the number of iterations without improvement to the best found solution
                           after which the algorithm should stop
        :param mutation_threshold: probability that violating the total night shift constraint is allowed
                                   when applying the move operator
        :param input_dir: directory with problem instances
        :param problem_instance: the number of problem instance
        """
        self._population_size = population_size
        self._best_candidates_fraction = best_candidates_fraction
        self._stop_after = stop_after
        self._mutation_threshold = mutation_threshold
        self._problem_instance = problem_instance
        self._forced_day_off = (
            pd.read_csv(f"{input_dir}/plan_{self._problem_instance}/forced_day_off.csv")
            .iloc[:, 1:]
            .values
        )
        self._pref_day_off = (
            pd.read_csv(f"{input_dir}/plan_{self._problem_instance}/pref_day_off.csv")
            .iloc[:, 1:]
            .values
        )
        pref_work_shift = (
            pd.read_csv(
                f"{input_dir}/plan_{self._problem_instance}/pref_work_shift.csv"
            )
            .iloc[:, 1:]
            .values
        )
        self._pref_work_shift = np.zeros(
            (self._forced_day_off.shape[0], self._forced_day_off.shape[1], 2)
        )
        self._pref_work_shift[:, :, 0] = (pref_work_shift == 1).astype(int)
        self._pref_work_shift[:, :, 1] = (pref_work_shift == 2).astype(int)
        self._qualified_route = (
            pd.read_csv(
                f"{input_dir}/plan_{self._problem_instance}/qualified_route.csv"
            )
            .iloc[:, 1:]
            .values
        )
        self.solution = None
        self.best_fitness_history = None
        self.best_fitness = None

    def run(self, random_seed: int = 0,) -> None:
        """
        Run the Genetic Algorithm with a given random seed
        :param random_seed:
        :return:
        """
        # set random seed
        np.random.seed(seed=random_seed)

        nr_of_couriers = self._forced_day_off.shape[0]
        nr_of_days = self._forced_day_off.shape[1]
        nr_of_routes = self._qualified_route.shape[1]

        best_candidates_number = round(
            self._best_candidates_fraction * self._population_size
        )

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
                    least_free = np.random.choice(least_qualified[days_off == days_off.max()])
                    # assign one of the feasible couriers if any
                    if least_free.size > 0:
                        solution[least_free, j, k, l] = 1
                    # otherwise a random one
                    else:
                        solution[np.random.choice(np.arange(11)), j, k, l] = 1
        initial_fitness = self._fitness(solution)

        # initial population
        population = [solution for x in range(self._population_size)]
        population_fitness = [initial_fitness for x in range(self._population_size)]

        best_fitness_history = [initial_fitness]

        # iterate until the stopping criterion is met
        while True:
            # apply move operator
            move_population = [self._move(copy.deepcopy(x)) for x in population]

            # apply crossover operator
            crossover_population = [
                self._crossover(copy.deepcopy(move_population))
                for x in range(int(len(move_population) / 2))
            ]
            crossover_population = list(itertools.chain(*crossover_population))

            # evaluate the candidates for the new population
            new_population_candidates = (
                population + move_population + crossover_population
            )

            candidates_fitness = population_fitness + [
                self._fitness(x) for x in move_population + crossover_population
            ]

            # select the top best_candidates_number of candidate solutions for the new population
            best_candidates_ids = sorted(
                range(len(candidates_fitness)), key=lambda i: candidates_fitness[i]
            )[-best_candidates_number:]

            # select the rest of the new population at random
            random_candidates_ids = list(
                np.random.choice(
                    [
                        x
                        for x in range(len(candidates_fitness))
                        if x not in best_candidates_ids
                    ],
                    self._population_size - best_candidates_number,
                )
            )

            # form the new population
            population = [
                new_population_candidates[x]
                for x in best_candidates_ids + random_candidates_ids
            ]
            population_fitness = [
                candidates_fitness[x]
                for x in best_candidates_ids + random_candidates_ids
            ]

            best_fitness_history += [max(candidates_fitness)]

            # check for stopping criterion and output the best solution if it's met
            if (
                best_fitness_history[-1]
                == best_fitness_history[
                    -min(self._stop_after, len(best_fitness_history))
                ]
            ):

                self.best_fitness_history = best_fitness_history
                self.best_fitness = best_fitness_history[-1]

                best_found_solution = new_population_candidates[
                    candidates_fitness.index(self.best_fitness)
                ]

                # output solution in the required format
                best_found_schedule = [
                    [i + 1, f"route{j+1}", f"day{k+1}", l + 1]
                    for i in range(nr_of_couriers)
                    for j in range(nr_of_routes)
                    for k in range(nr_of_days)
                    for l in range(2)
                    if best_found_solution[i, j, k, l] == 1
                ]
                self.solution = pd.DataFrame.from_dict(
                    {
                        "courier_id": [x[0] for x in best_found_schedule],
                        "day": [x[2] for x in best_found_schedule],
                        "rout_id": [x[1] for x in best_found_schedule],
                        "shift_id": [x[3] for x in best_found_schedule],
                    }
                )

                self.solution.to_csv(
                    f"ga_solution_{self._problem_instance}.csv", index=False
                )
                break

        return None

    def _fitness(self, solution: np.ndarray) -> float:
        """
        Calculate the fitness (objective value) of a given solution
        :param solution:
        :return: corresponding fitness value
        """
        fitness = 0

        # benefits for days off preferences
        fitness += 4 * (self._pref_day_off * (1 - solution.sum(axis=(1, 3)))).sum()

        # benefits for shift preferences
        fitness += 3 * (self._pref_work_shift * solution.sum(axis=1)).sum()

        # penalties for day shift following a night shift
        fitness -= (
            20
            * (
                solution[:, :, :-1, 1].sum(axis=1) * solution[:, :, 1:, 0].sum(axis=1)
            ).sum()
        )

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

    def _move(self, solution: np.ndarray) -> np.ndarray:
        """
        Apply the move operator to a given solution.
        First, a random shift is picked to be reassigned.
        Then, a random feasible courier is chosen to be assigned to it.
        Two types of feasible couriers are considered:
            1. Those that are not assigned to any shift on the given day
            2. Those that are already assigned to a shift on the given day.
               In that case they are exchanging shifts with the courier scheduled for the original shift.
        :param solution: solution to apply the move operator to
        :return: new solution
        """
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
            j, k, l, solution, ignore_night_shifts=False, consider_unassigned=False
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

    def _crossover(self, population: List) -> List:
        """
        Apply the crossover operator.
        First, two different parent solutions are chosen from the population at random.
        Then a day is chosen at random.
        The two offspring solutions are produced as follows:
            1. First part (before the chosen day) of the schedule of parent 1 is combined with
            the second part (starting from the chosen day) of the schedule of parent 2
            2. First part of the schedule of parent 2 is combined with
            the second part of the schedule of parent 1
        The only constraint that can get violated this way is the maximum number of night shifts per courier.
        :param population:
        :return: list with the two offspring solutions
        """

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
        j: int,
        k: int,
        l: int,
        solution: np.ndarray,
        ignore_night_shifts: bool = True,
        consider_unassigned: bool = True,
    ) -> np.ndarray:
        """
        Find the couriers that are feasible to be assigned
        to a given shift on a given route on a given day.
        Allows to consider either couriers that are unassigned yet
        or those already assigned.
        It is also an option to choose to ignore
        the constraint on the total number of night shifts.

        :param j: route id
        :param k: day id
        :param l: shift id
        :param solution:
        :param ignore_night_shifts: whether to ignore the night shift constraint or not
        :param consider_only_unassigned: whether to consider unassigned couriers
        :return: an array of feasible couriers
        """
        # couriers qualified for the given route
        qualified = np.where(self._qualified_route[:, j] == 1)[0]
        # couriers that do not have forced day off on the given day
        free = np.where(self._forced_day_off[:, k] == 0)[0]

        if not ignore_night_shifts:
            mutation_trigger = np.random.uniform()
            if mutation_trigger < self._mutation_threshold:
                ignore_night_shifts = True

        if consider_unassigned and not ignore_night_shifts and l == 1:
            not_scheduled_yet = np.where(solution[:, :, k, :].sum(axis=(1, 2)) == 0)[0]
            max_3_night_shifts = np.where(solution[:, :, :, 1].sum(axis=(1, 2)) < 4)[0]
            feasible_couriers = np.intersect1d(
                np.intersect1d(np.intersect1d(qualified, free), not_scheduled_yet),
                max_3_night_shifts,
            )
        elif consider_unassigned:
            not_scheduled_yet = np.where(solution[:, :, k, :].sum(axis=(1, 2)) == 0)[0]
            feasible_couriers = np.intersect1d(
                np.intersect1d(qualified, free), not_scheduled_yet
            )
        elif not ignore_night_shifts and l == 1:
            scheduled_for_day_shift = np.where(solution[:, :, k, 0].sum(axis=1) == 1)[0]
            scheduled_for_night_shift = np.where(solution[:, :, k, 1].sum(axis=1) == 1)[
                0
            ]
            max_3_night_shifts = np.where(solution[:, :, :, 1].sum(axis=(1, 2)) < 4)[0]
            max_4_night_shifts = np.where(solution[:, :, :, 1].sum(axis=(1, 2)) < 5)[0]
            feasible_couriers = np.concatenate(
                (
                    np.intersect1d(
                        np.intersect1d(
                            np.intersect1d(qualified, free), scheduled_for_day_shift
                        ),
                        max_3_night_shifts,
                    ),
                    np.intersect1d(
                        np.intersect1d(
                            np.intersect1d(qualified, free), scheduled_for_night_shift
                        ),
                        max_4_night_shifts,
                    ),
                )
            )
        else:
            scheduled = np.where(solution[:, :, k, :].sum(axis=(1, 2)) == 1)[0]
            feasible_couriers = np.intersect1d(
                np.intersect1d(qualified, free), scheduled
            )

        return feasible_couriers

    def _subarray_count(self, A, B) -> int:
        """
        Number of times array B is a subarray of A
        :param A:
        :param B:
        :return:
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

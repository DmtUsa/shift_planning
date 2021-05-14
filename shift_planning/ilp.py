from typing import Dict, List

import numpy as np
import pandas as pd
import pulp


class ILP:
    """
    Integer Linear Program for Shift Planning
    """

    def __init__(
        self, input_dir: str = "./input_data/", problem_instance: int = 1,
    ):
        self._problem_instance = problem_instance
        self._forced_day_off = pd.read_csv(
            f"{input_dir}/plan_{self._problem_instance}/forced_day_off.csv"
        )
        self._pref_day_off = pd.read_csv(
            f"{input_dir}/plan_{self._problem_instance}/pref_day_off.csv"
        )
        self._pref_work_shift = pd.read_csv(
            f"{input_dir}/plan_{self._problem_instance}/pref_work_shift.csv"
        )
        self._qualified_route = pd.read_csv(
            f"{input_dir}/plan_{self._problem_instance}/qualified_route.csv"
        )
        self.solution = None

    def solve(self) -> None:

        opt_model = pulp.LpProblem(name="ShiftPlanning")

        nr_of_couriers = self._forced_day_off.shape[0]
        nr_of_days = self._forced_day_off.shape[1] - 1
        nr_of_routes = self._qualified_route.shape[1] - 1

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

        # preferred days off
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
            for k in range(nr_of_days - 1)
        }

        # decision variables z_i_k
        # indicating if a courier i is assigned to three consecutive night shifts starting on day k
        z = {
            (i, k): pulp.LpVariable(cat=pulp.LpBinary, name=f"z_{i}_{k}")
            for i in C
            for k in range(nr_of_days - 3)
        }

        # Each route needs two couriers every day
        constraints_1 = {
            (j, k, l): opt_model.addConstraint(
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
            (i, k): opt_model.addConstraint(
                pulp.LpConstraint(
                    e=pulp.lpSum(x[i, j, k, l] for j in R for l in S),
                    sense=pulp.LpConstraintLE,
                    rhs=1,
                    name=f"2_constraint_{i}_{k}",
                )
            )
            for i in C
            for k in D
        }

        # Courier can be assigned to a route only if qualified
        constraints_3 = {
            (i, j): opt_model.addConstraint(
                pulp.LpConstraint(
                    e=pulp.lpSum(x[i, j, k, l] * (1 - q[i, j]) for k in D for l in S),
                    sense=pulp.LpConstraintEQ,
                    rhs=0,
                    name=f"3_constraint_{i}_{j}",
                )
            )
            for i in C
            for j in R
        }

        # Respect vacation
        constraints_4 = {
            (i, k): opt_model.addConstraint(
                pulp.LpConstraint(
                    e=pulp.lpSum(x[i, j, k, l] * v[i, k] for j in R for l in S),
                    sense=pulp.LpConstraintEQ,
                    rhs=0,
                    name=f"4_constraint_{i}_{k}",
                )
            )
            for i in C
            for k in D
        }

        # Not more than 4 night shifts
        constraints_5 = {
            i: opt_model.addConstraint(
                pulp.LpConstraint(
                    e=pulp.lpSum(x[i, j, k, 2] for j in R for k in D),
                    sense=pulp.LpConstraintLE,
                    rhs=4,
                    name=f"5_constraint_{i}",
                )
            )
            for i in C
        }

        # Dummy variable y_i_k definition - part 1
        constraints_6 = {
            (i, k): opt_model.addConstraint(
                pulp.LpConstraint(
                    e=2 * y[i, k]
                    - pulp.lpSum((x[i, j, k, 2] + x[i, j, k + 1, 1]) for j in R),
                    sense=pulp.LpConstraintLE,
                    rhs=0,
                    name=f"6_constraint_{i}_{k}",
                )
            )
            for i in C
            for k in range(nr_of_days - 1)
        }

        # Dummy variable y_i_k definition - part 2
        constraints_7 = {
            (i, k): opt_model.addConstraint(
                pulp.LpConstraint(
                    e=pulp.lpSum((x[i, j, k, 2] + x[i, j, k + 1, 1]) for j in R)
                    - y[i, k],
                    sense=pulp.LpConstraintLE,
                    rhs=1,
                    name=f"7_constraint_{i}_{k}",
                )
            )
            for i in C
            for k in range(nr_of_days - 1)
        }

        # Dummy variable z_i_k definition - part 1
        constraints_8 = {
            (i, k): opt_model.addConstraint(
                pulp.LpConstraint(
                    e=4 * z[i, k]
                    - pulp.lpSum(
                        (
                            x[i, j, k, 2]
                            + x[i, j, k + 1, 2]
                            + x[i, j, k + 2, 2]
                            + x[i, j, k + 3, 2]
                        )
                        for j in R
                    ),
                    sense=pulp.LpConstraintLE,
                    rhs=0,
                    name=f"8_constraint_{i}_{k}",
                )
            )
            for i in C
            for k in range(nr_of_days - 3)
        }

        # Dummy variable z_i_k definition - part 2
        constraints_9 = {
            (i, k): opt_model.addConstraint(
                pulp.LpConstraint(
                    e=pulp.lpSum(
                        (
                            x[i, j, k, 2]
                            + x[i, j, k + 1, 2]
                            + x[i, j, k + 2, 2]
                            + x[i, j, k + 3, 2]
                        )
                        for j in R
                    )
                    - z[i, k],
                    sense=pulp.LpConstraintLE,
                    rhs=3,
                    name=f"9_constraint_{i}_{k}",
                )
            )
            for i in C
            for k in range(nr_of_days - 3)
        }

        # add objective
        objective = (
            4
            * pulp.lpSum(
                (1 - pulp.lpSum(x[i, j, k, l] for j in R for l in S)) * d[i, k]
                for i in C
                for k in D
            )
            + 3
            * pulp.lpSum(
                x[i, j, k, l] * s[i, k, l] for i in C for j in R for k in D for l in S
            )
            - 20 * pulp.lpSum(y[i, k] for i in C for k in range(nr_of_days - 1))
            - 10 * pulp.lpSum(z[i, k] for i in C for k in range(nr_of_days - 3))
        )

        # set to maximize the objective
        opt_model.sense = pulp.LpMaximize
        opt_model.setObjective(objective)

        # solving with CBC solver
        pulp.LpSolverDefault.msg = 1
        opt_model.solve()

        # extract solution
        solution = [
            {
                "courier_id": i + 1,
                "day": f"day{k + 1}",
                "rout_id": f"route{j + 1}",
                "shift_id": l,
            }
            for i in C
            for j in R
            for k in D
            for l in S
            if (x[i, j, k, l].varValue == 1)
        ]

        # output in GA format
        self.solution = np.array(
            [[[[x[i, j, k, l].varValue for l in S] for k in D] for j in R] for i in C]
        )

        # save solution in the required format
        self._solution = pd.DataFrame.from_dict(
            {
                "courier_id": [x["courier_id"] for x in solution],
                "day": [x["day"] for x in solution],
                "rout_id": [x["rout_id"] for x in solution],
                "shift_id": [x["shift_id"] for x in solution],
            }
        )

        self._solution.to_csv(f"ilp_solution_{self._problem_instance}.csv", index=False)

        return None

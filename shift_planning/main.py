from shift_planning.metaheuristics import GeneticAlgorithm
from shift_planning.milp import MILP

if __name__ == "__main__":
    milp = MILP()
    milp.solve()
    ga = GeneticAlgorithm()
    print(ga._forced_day_off)

from shift_planning.ilp import ILP
from shift_planning.metaheuristics import GeneticAlgorithm
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Solve the ILP formulation
    ilp = ILP()
    ilp.solve()

    # Run Genetic Algorithm
    ga = GeneticAlgorithm(ilp.solution)
    # ga = GeneticAlgorithm(None)
    ga.run(random_seed=3)

    # plot GA's incumbent solution fitness per iteration
    optimal_fitness = ga._fitness(ilp.solution)
    plt.plot(ga.best_fitness_history)
    plt.plot([optimal_fitness for x in range(len(ga.best_fitness_history))])
    plt.plot()
    plt.show()

    print("Optimal fitness:", optimal_fitness)
    print("Best fitness obtained by Genetic Algorithm:", ga.best_fitness)

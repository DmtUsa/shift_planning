from shift_planning.ilp import ILP
from shift_planning.metaheuristics import GeneticAlgorithm
from matplotlib import pyplot as plt
import time

if __name__ == "__main__":
    # Solve the ILP formulation
    ilp = ILP()
    ilp.solve()

    # Run Genetic Algorithm
    ga = GeneticAlgorithm(
        population_size=130,
        best_candidates_fraction=0.3,
        stop_after=15,
        mutation_threshold=0.05,
    )
    start_time = time.time()
    ga.run(random_seed=3)

    total_time = round(time.time() - start_time)

    # plot GA's incumbent solution fitness per iteration
    optimal_fitness = ga._fitness(ilp.solution)
    plt.plot(ga.best_fitness_history)
    plt.plot([optimal_fitness for x in range(len(ga.best_fitness_history))])
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend([f"GA: {int(ga.best_fitness)}", f"Optimal: {int(optimal_fitness)}"])
    plt.show()

    print(f"Optimal fitness: {optimal_fitness}")
    print(f"Best fitness obtained by Genetic Algorithm: {ga.best_fitness}")
    print(f"GA run time: {total_time} seconds")

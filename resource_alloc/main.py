from typing import List
import random
import numpy as np
from deap import base, creator, tools, algorithms
from alloc_scenario import ResourceAllocScenario, TaskFunction, get_max_duration, simple_discount_length_func, simple_robust_n_discount_length_func

N_RESOURCES = 4
TASKS = [
    TaskFunction(size=10, length_func=simple_robust_n_discount_length_func),
    TaskFunction(size=5, length_func=simple_robust_n_discount_length_func),
    TaskFunction(size=2, length_func=simple_robust_n_discount_length_func),
]
TASKS = [
    TaskFunction(size=10, length_func=simple_discount_length_func),
    TaskFunction(size=5, length_func=simple_discount_length_func),
    TaskFunction(size=2, length_func=simple_discount_length_func),
]

N_TASKS = len(TASKS)


# Define the fitness function
def eval_fitness(resource_alloc_vector: List[int]) -> float:

    scenario = ResourceAllocScenario(n_resources=N_RESOURCES, tasks=TASKS)

    return get_max_duration(resource_alloc_vector=resource_alloc_vector, scenario=scenario),


# Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=N_TASKS * N_RESOURCES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create initial population and run the genetic algorithm
population = toolbox.population(n=50)
result = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=200, verbose=False)

best_ind = tools.selBest(population, k=1)[0]
print("Best individual is %s, with fitness %s" % (best_ind, best_ind.fitness.values))
print(np.reshape(best_ind, (N_TASKS, N_RESOURCES)))
# [
# [1, 0, 1, 0],
# [0, 1, 0, 1],
# [1, 1, 1, 1]
# ]

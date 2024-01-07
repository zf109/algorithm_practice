import random
from deap import base, creator, tools, algorithms

# Define the fitness function
def evalFitness(individual):
    target = 50
    return abs(sum(individual) - target),

# Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalFitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create initial population and run the genetic algorithm
population = toolbox.population(n=50)
result = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=False)

best_ind = tools.selBest(population, k=1)[0]
print("Best individual is %s, with fitness %s" % (best_ind, best_ind.fitness.values))

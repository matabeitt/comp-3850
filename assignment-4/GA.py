import numpy as np


def fitness(f, x):
    """
    Supplied function f(x) returns a value for fitness so long as f(x) has a range >= 0
    :param f:
    :param x:
    :return:
    """
    # return np.exp(f(x))
    # e^y made table unreadable from extremely small numbers
    # return f(x) if (int(x,2) > 0 and int(x,2) < 15) else 1
    return f(x) if f(x) > 0 else 1


def generate_population(size):
    pop = ['{0:04b}'.format(i) for i in range(size)]
    return pop


def choose(population, size, relative=None):
    if relative is None:
        return np.random.choice(population, size)
    return np.random.choice(population, relative, size)


def choose_parents(group, relative):
    x1, x2 = np.random.choice(a=group, p=relative, size=2)
    return x1, x2


def crossover(a, b):
    pivot = np.random.randint(0, max(len(a), len(b)), size=1)[0]
    c1 = a[0:pivot] + b[pivot:len(b)]
    c2 = b[0:pivot] + a[pivot:len(a)]
    return c1, c2


def mutate(child):
    pivot = np.random.randint(0, len(child))

    if child[pivot] == '0':
        r = '1'
    else:
        r = '0'

    child = child[:pivot] + r + child[pivot+1:]
    return child


def ga(func, pop_size=15, elite_size=6, generations=500, pc=0.7, pm=0.1):
    mostfit = -1
    fitnesses = None
    relative = None
    avgfits = []

    population = generate_population (pop_size)

    for epoch in range(1, generations):
        print("===========================")
        print("Generation", epoch)
        print("---------------------------")
        elites = choose(population, elite_size)

        while elites.size < pop_size:
            fitnesses = [fitness(func, elite) for elite in elites]
            relative = [(x/sum(fitnesses)) for x in fitnesses]
            x1, x2 = choose(elites, relative, 2)

            if np.random.random() <= pc:
                c1, c2 = crossover(x1, x2)
                if np.random.random() <= pm:
                    c1 = mutate(c1)
                    fitnesses.append(fitness(func, c1))
                    elites = np.append(elites, c1)
                if np.random.random() <= pm:
                    c2 = mutate(c2)
                    fitnesses.append(fitness(func, c1))
                    elites = np.append(elites, c2)

        avgfits.append(np.mean(fitnesses))
        population = elites
        print("Average Fitness:", np.around(np.mean(fitnesses), decimals=3))
        print("{:>2}{:>10}{:>10}{:>15}{:>15}".format("N", "bin", "int", "fitness", "relative"))
        for i in range(len(elites)):
            print("{:>2}{:>10}{:>10}{:>15.4g}{:>15.2g}".format(
                i, elites[i], int(elites[i], 2), np.around(fitness(func, elites[i]), decimals=4),
                fitnesses[i]/sum(fitnesses)))
        print("===========================")
    return population, avgfits

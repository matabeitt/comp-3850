import numpy as np


def initialize(inputs):
    weights = np.random.uniform(-0.5, 0.5, size=len(inputs[0]))
    bias = np.random.uniform(-0.5, 0.5, size=1)[0]

    return weights, bias


def activation(x, weights, bias):
    y = np.dot(x, weights) - bias
    y = 1 if y >= 0 else 0
    return y


def train(inputs, outputs, weights, bias, a=0.1):
    error = [1, 1, 1, 1]
    iterations = 0

    while error != [0, 0, 0, 0]:
        print("=========================================================")
        print("{0}{1:>8}{2:>9}{3:>9}{4:>4}{5:>4}{6:>9}{7:>9}"
              .format("X", "Yd", "W1_0", "W2_0", "Ya", "E", "W1_1", "W2_1"))
        for i in range(len(inputs)):
            y = activation(inputs[i], weights, bias)

            error[i] = outputs[i] - y

            delta = a * np.dot(inputs[i], error[i])

            old_weights = weights
            weights = weights + delta

            print("{0}{1:>4}{2:>9}{3:>9}{4:>4}{5:>4}{6:>9}{7:>9}"
                  .format(inputs[i], outputs[i],
                          np.around(old_weights[0], decimals=3), np.around(old_weights[1], decimals=3),
                          y, error[i],
                          np.around(weights[0], decimals=3), np.around(weights[1], decimals=3)))

            if iterations >= 200000 and i == 3:
                return False, iterations

        iterations = iterations + 1

    return True, iterations

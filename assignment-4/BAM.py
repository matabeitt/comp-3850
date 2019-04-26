import numpy as np

def train(pairs):
    x = len(pairs[0][0])
    y = len(pairs[0][1])
    W = np.zeros((x, y))
    for x,y in pairs:
        W += np.dot(x,y.T)
    return W

def test(weight, pairs):
    for x, y in pairs:
        test = np.sign(np.dot(weight.T, x))
        if np.array_equal(y, test) is False:
            return False

    return True

def recall(weight, pairs, probe):
    X = probe
    while not stable(X, pairs):
        Y = np.sign(np.dot(weight.T, X))
        X = np.sign(np.dot(weight, Y))
    return X


def stable(arr, fundamental):
    for x,y in fundamental:
        if np.array_equal(arr, x) or np.array_equal(arr, y):
            return True
    return False


def bam(pairs, probe):

    weight = train(pairs)
    print("Weight Matrix", weight)
    stable = test(weight, pairs)
    print("Is stable?", stable)
    if not stable: return False
    recalled = recall(weight, pairs, probe)
    print(probe, "recalled matrix", recalled)
    return True

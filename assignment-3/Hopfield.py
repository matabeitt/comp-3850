import numpy as np


def stable(state, fundamentals):
    for y in fundamentals:
        if np.array_equal(y, state): return True
    return False


def train(states):

    M = len(states)
    col = len(states[0])
    I = np.identity(col)
    W = np.zeros((col,col))

    for Y in states:
        Yt = np.transpose(Y)
        W = W + np.multiply(Y, Yt)

    W = W - np.multiply(M, I)
    return W


def activate (W, states):
    Y = []
    for X in states:
        o = np.dot(W, X)
        o = np.vectorize(lambda x: -1 if x<0 else 1)(o)
        Y.append(o)
    return Y


def recall(W, fundamental, states):
    recalled = []

    for x in states:
        y = None
        while not stable(y, fundamental):
            y = np.dot(W, x)
            for i in range(y.size):
                if y[i] > 1:
                    y[i]=1
                elif y[i] < -1:
                    y[i]=-1
            x = y
        recalled.append(y)

    return recalled


def network(fundamental, probes):
    W = train(fundamental)
    S = activate(W, fundamental)
    if not np.array_equal(S, fundamental):
        return False
    Y = recall(W, fundamental, probes)
    return Y
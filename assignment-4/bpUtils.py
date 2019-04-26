import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))
    a = 1.716
    b = 0.667
    return ((2*a) / (1 + np.exp(-x*b)))-a


def gradient(x):
    return x * (1 - x)


def initialize(inputs, nodes):

    weights = np.random.uniform(-2.4/len(inputs[0]), 2.4/len(inputs[0]), size=len(inputs[0])*nodes)
    biases = np.random.uniform(-2.4/len(inputs[0]), 2.4/len(inputs[0]), size=nodes)

    return weights, biases


def activate(x, wh, bh, wo, bo):
    yh = bh
    yh
    for i in range(len(yh)):
        weightsum = 0
        for k in range(len(x)):
            j = k * len(x) + i
            weightsum += x[i] * wh[j]
        yh[i] += weightsum
    yh = sigmoid(yh)
    yo = bo
    yo += yh[0] * wo[0] + yh[1] * wo[1]
    yo = sigmoid(yo)
    return yo, yh


def backpropogation(inputs, outputs, wh, bh, wo, bo, a=0.1):
    error = [0,0,0,0]
    for l in range(len(inputs)):
        ya, yh = activate(inputs[l], wh, bh, wo, bo)

        x = inputs[l]
        yd = outputs[l]

        error[l] = yd - ya  # 1x1

        # gradient_k = y_k * (1 - y_k) * e_k
        dk = gradient(ya) * error[l]

        # delta_w_jk = a * y_j * gradient_k
        dw_jk = [0, 0]
        for j in range(len(dw_jk)):
            dw_jk[j] = a * yh[j] * dk

        # w_jk = w_jk + delta_w_jk
        for jk in range(len(wo)):
            wo[jk] = wo[jk] + dw_jk[jk]

        # gradient_j = y_j * (1 - y_j) * sumall(gradient_k * w_jk)
        dj = gradient(yh)
        for i in range(len(dj)):
            dj[i] = dj[i] * wo[i] * dk

        # delta_w_ij = a * x_i * gradient_j
        dw_ij = [0, 0, 0, 0]
        # x1 -> w1, w2 -> h1
        # x2 -> w3, w4 -> h2
        for i in range(len(dw_ij)):
            k = int(i/2)
            j = i%2
            dw_ij[i] = a * x[k] * dj[j]

        print("{0}{1}{2}{3}".format(x, yd, ya, error[l]))


    sse = 0
    for i in range(len(error)):
        sse += np.power(error[i], 2)
    print("SSE: ", sse)

    return sse


def train(inputs, outputs, hidden_weights, hidden_bias, output_weights, output_bias):
    sse = 1
    while sse > 0.001:
        print("X\t\tYa\t\t\tE")
        sse = backpropogation(inputs, outputs, hidden_weights, hidden_bias, output_weights, output_bias)
        print("\n\n")
    return sse

import GA
import Hopfield
import numpy as np


# RUN GENETIC ALGORITHM
def absf(x):
    x = int(x, 2)
    return 15*x - pow(x, 2)


result, avgs = GA.ga(absf, 15, 6, 500)
print([int(x,2) for x in result])

# RUN HOPFIELD NETWORK
fundamental = [np.array([1, 1, 1]), np.array([-1, -1, -1])]
probes = [np.array([1, -1, 1]), np.array([-1, 1, -1])]
result = Hopfield.network(fundamental, probes)
print("Input probes", probes, "recalled a result", result)


# RUN BAM
x = np.vstack([1,1,1,1,1,1])
y = np.vstack([1,1,1])

x1 = np.vstack([-1,-1,-1,-1,-1,-1])
y1 = np.vstack([-1,-1,-1])

x2 = np.vstack([1,1,-1,-1,1,1])
y2 = np.vstack([1,-1,1])

x3 = np.vstack([-1,-1,1,1,-1,-1])
y3 = np.vstack([-1,1,-1])

probe = np.vstack([-1,1,1,1,1,1])

print(BAM.bam([(x, y), (x1, y1), (x2,y2), (x3,y3)], probe))
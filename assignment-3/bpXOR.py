import bpUtils as BP

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]
Wh, Bh = BP.initialize(X, 2)
Wo, Bo = BP.initialize([Bh], 1)
BP.train(X, Y, Wh, Bh, Wo, Bo)

"""
Attempted to write the BPXOR algorithm using vectors
and matrices so it could be dynamic to the number of 
inputs but then i got frustrated and gave up.
"""


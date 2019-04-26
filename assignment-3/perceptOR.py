import percepUtils as perceptron

X = [[0,0], [0,1], [1,0], [1,1]]
Y = [0, 1, 1, 1]


W, B = perceptron.initialize(X)
success, epochs = perceptron.train(X, Y, W, B)

print("Unable to solve OR in", epochs, "epochs.") if success is False else print("Model trained OR in", epochs, "epochs.")

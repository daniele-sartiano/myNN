import sys
import numpy as np

class MyNN(object):
    def __init__(self, seed=7):
        np.random.seed(seed)
        self.seed = seed
        self.weights = 2 * np.random.random((3, 1)) -1

    def sigmoid(self, x):
        """ Normalize between 0 and 1"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """ Gradient of the sigmoid curve: how confident we are about the existing weights."""
        return x * (1-x)
        
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.predict(X)
            error = y - output
            adj = np.dot(X.T, error * self.sigmoid_derivative(output))
            self.weights += adj

    def predict(self, X):
        return self.sigmoid(np.dot(X, self.weights))

epochs = int(sys.argv[1])

nn = MyNN()

X = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
y = np.array([[0,1,1,0]]).T

nn.train(X, y, epochs)

print(nn.predict([1,0,1]))
print(nn.predict([0,0,1]))
print(nn.predict([1,1,1]))

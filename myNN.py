import sys
import numpy as np

class MyNN(object):
    def __init__(self, seed=7):
        np.random.seed(seed)
        self.seed = seed
        self.weights = 2 * np.random.random((3, 1)) -1

    def train(self, X, y, epochs):
        for epoch in xrange(epochs):
            output = self.predict(X)
            error = y - output
            adj = np.dot(X.T, error * (output * (1-output)))
            self.weights += adj
            # if (epoch % 1000 == 0):
            #     print "Error", error

    def predict(self, X):
        return 1/(1+np.exp(-np.dot(X, self.weights)))

epochs = int(sys.argv[1])

nn = MyNN()

X = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
y = np.array([[0,1,1,0]]).T

nn.train(X, y, epochs)

print nn.predict([1,0,1])

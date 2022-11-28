import numpy as np

class SimplePerceptron:
    learn_rate = 0.1

    def __init__(self):
        self.weights = None
        
    def logistic_function(self, x: float) -> float:
        """
        Logistic function, used as the activation function
        """
        return 1. / (1 + np.exp(-x))

    def forward_pass(self, X: np.ndarray) -> float:
        """
        Prediction of a single data point, given the current weights
        """
        weighted_sum = np.dot(X, self.weights)
        output = self.logistic_function(weighted_sum)

        return output

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_epochs: int = 20):
        """
        Training using Batch Gradient Descent.
        Weights array is updated after each epoch
        """
        self.weights = np.random.uniform(-1, 1, X_train.shape[1])
        current_weights = self.weights.copy()
        for epoch in range(n_epochs):
            for x, y in zip(X_train, y_train):
                y_predicted = self.forward_pass(x)
                current_weights -= self.learn_rate * (y_predicted - y) * y_predicted * (1 - y_predicted) * x
            self.weights = current_weights.copy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict label for unseen data
        """
        return np.array([self.forward_pass(x) for x in X_test])
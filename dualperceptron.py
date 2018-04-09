import numpy as np
from math import *

class DualPerceptron:
    def __init__(self,kernel):
        self.kernel = kernel

    def fit(self,X,y):

        self.w = np.array([0] * len(X[0]))
        self.alpha = np.array([0] * len(X))
        for i in range(len(X)):
            y_hat = 0
            for j in range(len(X)):
                if self.kernel == "linear":
                    y_hat = y_hat + self.alpha[j] * y[j] * self.linear_kernel(X[j],X[i])
                elif self.kernel == "rbf":
                    y_hat = y_hat + self.alpha[j] * y[j] * self.rbf(X[j], X[i])
                else:
                    y_hat = y_hat + self.alpha[j] * y[j] * np.dot(X[j], X[i])
            if y_hat >= 0:
                y_hat = 1
            else:
                y_hat = -1
            if y[i] != y_hat :
                self.alpha[i] = self.alpha[i] + 1
        for k in range(len(X)):
            self.w = self.w + self.alpha[k] * y[k] * X[k]

    def fit_linear_kernel(self,X,y):
        self.w = np.array([0] * len(X[0]))
        self.alpha = np.array([0] * len(X))
        for i in range(len(X)):
            y_hat = 0
            for j in range(len(X)):
                y_hat = y_hat + self.alpha[j] * y[j] * self.linear_kernel(X[j],X[i])
            if y_hat >= 0:
                y_hat = 1
            else:
                y_hat = -1
            if y[i] != y_hat :
                self.alpha[i] = self.alpha[i] + 1
        for k in range(len(X)):
            self.w = self.w + self.alpha[k] * y[k] * X[k]

    #model prediction for the given feature data X
    def predict(self, X):
        #if self.kernel == "linear":
        #    final_scores = np.array([self.linear_kernel(self.w.T,x) for x in X])
        #elif self.kernel == "rbf":
        #    final_scores = np.array([self.rbf(self.w.T, x) for x in X])
        #else:
        #    final_scores = np.array([np.dot(self.w.T, x) for x in X])
        final_scores = np.array([np.dot(self.w.T, x) for x in X])
        preds = [1 if x >= 0.0 else -1 for x in final_scores]
        return preds

    def predict_new(self,X_test,X_train,y_train,alpha):
        preds = []
        for i in range(len(X_test)):
            sum = 0
            for j in range(len(alpha)):
                sum += alpha[j] * y_train[j] * self.rbf(X_test[i],X_train[j])
            if sum >= 0.0:
                pred = 1
            else:
                pred = -1
            preds.append(pred)
        return preds

    @staticmethod
    # Calculate accuracy percentage
    def linear_kernel(x,y):
        return np.dot(x,y)

    @staticmethod
    def rbf(va, vb):
        gamma = 0.15
        temp = va - vb
        return exp(-gamma * np.sum(np.dot(temp,temp)))
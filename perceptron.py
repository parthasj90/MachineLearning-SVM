import numpy as np
import sys
import math

class Perceptron:
    def __init__(self,learningrate,max_iterations):
        self.learningrate = learningrate
        self.max_iterations = max_iterations

    def fit(self,X,y):
        iter = 0
        mismatch_flag = 1
        while iter <= self.max_iterations and mismatch_flag != 0:
            mismatch_flag = 0
            self.w = np.array([0] * len(X[0]))
            for i in range(len(X)):
                y_hat = np.dot(self.w.T,X[i,:])
                if y_hat >= 0.0:
                    y_hat = 1
                else:
                    y_hat = -1
                if y[i]*y_hat <= 0:
                    mismatch_flag = 1
                    self.w = self.w + self.learningrate * y[i] * X[i]
            iter += 1


    #model prediction for the given feature data X
    def predict(self, X):
        final_scores = np.array([np.dot(self.w.T,x) for x in X])
        preds = [1 if x >= 0.0 else -1 for x in final_scores]
        return preds

    @staticmethod
    # Calculate accuracy percentage
    def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0
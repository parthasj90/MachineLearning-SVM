import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def my_model(X_train,y_train,X_test,y_test):
    from dualperceptron import DualPerceptron
    constant = np.ones((len(X_train),1))
    X_train = np.hstack((constant,X_train))
    const = np.ones((len(X_test), 1))
    X_test = np.hstack((const, X_test))
    lr = DualPerceptron("rbf")
    lr.fit(X_train,y_train)
    y_pred_test = lr.predict(X_test)
    from sklearn.metrics import precision_recall_fscore_support
    output = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')
    return output

def main():
    filename = 'twoSpirals.csv'
    dataset = pd.read_csv(filename,header=None)
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values

    from sklearn.utils import shuffle
    X,y = shuffle(X,y,random_state=13)
    #kfold generation
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    counter = 1
    kf = KFold(n_splits=10)
    allAccuracy = []
    allRecall = []
    for train_index,test_index in kf.split(X):
        print("Executing Fold ",counter)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        #print("Executing SKLearn model")
        #sk_model(X_train,y_train,X_test,y_test)
        print("Executing my model")
        output = my_model(X_train,y_train,X_test,y_test)
        allAccuracy.append(output[0])
        allRecall.append(output[1])
        print("Accuracy: ",output[0])
        print("Recall: ",output[1])
        counter += 1
    print("Average Accuracy across all folds",np.mean(allAccuracy))
    print("Standard deviation Accuracy across all folds", np.std(allAccuracy))
    print("Average Recall across all folds",np.mean(allRecall))
    print("Standard Deviation Accuracy across all folds", np.std(allRecall))


if __name__ == '__main__':
    main()

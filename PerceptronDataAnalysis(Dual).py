import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def my_model(X_train,y_train,X_test,y_test):
    from dualperceptron import DualPerceptron
    constant = np.ones((len(X_train),1))
    X_train = np.hstack((constant,X_train))
    const = np.ones((len(X_test), 1))
    X_test = np.hstack((const, X_test))
    lr = DualPerceptron(" ")
    lr.fit(X_train,y_train)
    y_pred_test = lr.predict_new(X_test,X_train,)
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test, average='micro')
    recall_test = recall_score(y_test, y_pred_test, average='micro')
    return acc_test,prec_test,recall_test

def main():
    filename = 'perceptronData.csv'
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
    allPrecision = []
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
        accuracy, precision, recall = my_model(X_train, y_train, X_test, y_test)

        allAccuracy.append(accuracy)
        allPrecision.append(precision)
        allRecall.append(recall)
        print("Accuracy: ", accuracy)
        print("Precision ", precision)
        print("Recall: ", recall)
        counter += 1
    print("Average Accuracy across all folds",np.mean(allAccuracy))
    print("Standard deviation Accuracy across all folds", np.std(allAccuracy))
    print("Average Precision across all folds",np.mean(allPrecision))
    print("Standard Deviation Precision across all folds", np.std(allPrecision))
    print("Average Recall across all folds",np.mean(allRecall))
    print("Standard Deviation Accuracy across all folds", np.std(allRecall))


if __name__ == '__main__':
    main()

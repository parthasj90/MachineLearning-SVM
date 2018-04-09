import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score , make_scorer
from sklearn.metrics import average_precision_score
def main():
    filename = 'diabetes.csv'
    dataset = pd.read_csv(filename,header=None)
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values

    from sklearn.utils import shuffle
    X,y = shuffle(X,y,random_state=13)
    #kfold generation
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler


    kernels  = ['rbf','linear']
    for kern in kernels:
        print("Executing kernel: ",kern)
        kf = KFold(n_splits=10)
        allTrainAccuracy = []
        allTrainRecall = []
        allTrainPrec = []
        allTestAccuracy = []
        allTestRecall = []
        allTestPrec = []
        counter = 1
        for train_index,test_index in kf.split(X):
            print("Executing Fold ",counter)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)

            #parameters = {'kernel': ('linear', 'rbf'), "C": [2**x for x in range(-5,11)], "gamma": [2**x for x in range(-15,6)]}
            if kern == "linear":
                parameters = {"C": [2 ** x for x in range(-5, 11)]}
            else:
                parameters = {"C": [2 ** x for x in range(-5, 11)], "gamma": [2**x for x in range(-15,6)]}
            svc = svm.SVC(kernel=kern)
            accuracy_scorer = make_scorer(accuracy_score)
            clf = GridSearchCV(svc, parameters,scoring=accuracy_scorer)
            clf.fit(X_train,y_train)
            print("The best parameters are %s with a score of %0.2f"% (clf.best_params_, clf.best_score_))
            predicted_test = clf.predict(X_test)
            predicted_train = clf.predict(X_train)
            #from sklearn.metrics import precision_recall_fscore_support
            #output = precision_recall_fscore_support(y_test, predicted, average='weighted')
            acc_train = accuracy_score(y_train, predicted_train)
            prec_train = average_precision_score(y_train, predicted_train)
            recall_train = recall_score(y_train, predicted_train)
            allTrainAccuracy.append(acc_train)
            allTrainPrec.append(prec_train)
            allTrainRecall.append(recall_train)
            print("Train Accuracy: ",acc_train)
            print("Train Precision: ",prec_train)
            print("Train Recall: ",recall_train)
            acc_test = accuracy_score(y_test, predicted_test)
            prec_test = average_precision_score(y_test, predicted_test)
            recall_test = recall_score(y_test, predicted_test)
            allTestAccuracy.append(acc_test)
            allTestPrec.append(prec_test)
            allTestRecall.append(recall_test)
            print("test Accuracy: ", acc_test)
            print("test Precision: ", prec_test)
            print("test Recall: ", recall_test)
            counter += 1
        print("TRAINING RESULTS")
        print("Average Accuracy across all folds",np.mean(allTrainAccuracy))
        print("Standard deviation Accuracy across all folds", np.std(allTrainAccuracy))
        print("Average Precision across all folds", np.mean(allTrainPrec))
        print("Standard deviation Precision across all folds", np.std(allTrainPrec))
        print("Average Recall across all folds",np.mean(allTrainRecall))
        print("Standard Deviation Accuracy across all folds", np.std(allTrainRecall))
        print("TESTING RESULTS")
        print("Average Accuracy across all folds",np.mean(allTestAccuracy))
        print("Standard deviation Accuracy across all folds", np.std(allTestAccuracy))
        print("Average Precision across all folds", np.mean(allTestPrec))
        print("Standard deviation Precision across all folds", np.std(allTestPrec))
        print("Average Recall across all folds",np.mean(allTestRecall))
        print("Standard Deviation Accuracy across all folds", np.std(allTestRecall))


if __name__ == '__main__':
    main()

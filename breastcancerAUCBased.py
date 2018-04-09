import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score , make_scorer,roc_auc_score
from sklearn.metrics import average_precision_score
def main():
    filename = 'breastcancer.csv'
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
        counter = 1
        kf = KFold(n_splits=10)
        allTrainROCAUC = []
        allTestROCAUC = []
        print("Executing kernel: ",kern)
        for train_index,test_index in kf.split(X):
            print("Executing Fold ",counter)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            if kern == "linear":
                parameters = {"C": [2 ** x for x in range(-5, 11)]}
            else:
                parameters = {"C": [2 ** x for x in range(-5, 11)], "gamma": [2**x for x in range(-15,6)]}
            svc = svm.SVC(kernel=kern)
            roc_auc_scorer = make_scorer(roc_auc_score)
            clf = GridSearchCV(svc, parameters,scoring=roc_auc_scorer)
            clf.fit(X_train,y_train)
            print("The best parameters are %s with a score of %0.2f"% (clf.best_params_, clf.best_score_))
            predicted_test = clf.predict(X_test)
            predicted_train = clf.predict(X_train)
            roc_auc_train = roc_auc_score(y_train, predicted_train)
            roc_auc_test = roc_auc_score(y_test, predicted_test)
            allTrainROCAUC.append(roc_auc_train)
            print("Train ROC AUC: ",roc_auc_train)
            allTestROCAUC.append(roc_auc_test)
            print("Test ROC AUC: ", roc_auc_test)
            counter += 1
        print("TRAINING RESULTS")
        print("Average ROC AUC across all folds",np.mean(allTrainROCAUC))
        print("Standard deviation ROC AUC across all folds", np.std(allTrainROCAUC))
        print("TESTING RESULTS")
        print("Average ROC AUC across all folds",np.mean(allTestROCAUC))
        print("Standard deviation ROC AUC across all folds", np.std(allTestROCAUC))
        plt.plot([x for x in range(1,11)], allTestROCAUC)
        plt.title('ROC AUC for all folds:'+ kern)
        plt.xlabel('FOLDS')
        plt.ylabel('ROC AUC')
        plt.show()

if __name__ == '__main__':
    main()

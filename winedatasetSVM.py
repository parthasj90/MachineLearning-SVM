import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score , make_scorer,roc_auc_score
from sklearn.metrics import precision_score
def main():
    filename = 'wine.data'
    dataset = pd.read_csv(filename,header=None)
    X = dataset.iloc[:,1:].values
    y = dataset.iloc[:,0].values

    from sklearn.utils import shuffle
    X,y = shuffle(X,y,random_state=13)
    y1 = np.array([1 if n == 1 else 0 for n in y])
    y2 = np.array([1 if n == 2 else 0 for n in y])
    y3 = np.array([1 if n == 3 else 0 for n in y])

    #kfold generation
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    kernels  = ['rbf','linear']
    for kern in kernels:
        kf = KFold(n_splits=10)
        allTrainAccuracy = []
        allTrainRecall = []
        allTrainPrec = []
        allTestAccuracy = []
        allTestRecall = []
        allTestPrec = []
        allTestROCAUC1 = []
        allTestROCAUC2 = []
        allTestROCAUC3 = []
        print("Executing kernel: ",kern)
        counter = 1
        for train_index,test_index in kf.split(X):
            print("Executing Fold ",counter)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train1, y_test1 = y1[train_index], y1[test_index]
            y_train2, y_test2 = y2[train_index], y2[test_index]
            y_train3, y_test3 = y3[train_index], y3[test_index]

            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            if kern == "linear":
                parameters = {"C": [2 ** x for x in range(-5, 11)]}
            else:
                parameters = {"C": [2 ** x for x in range(-5, 11)], "gamma": [2**x for x in range(-15,6)]}

            svc = svm.SVC(kernel=kern,probability=True)
            accuracy_scorer = make_scorer(accuracy_score)
            clf = GridSearchCV(svc, parameters,scoring=accuracy_scorer)

            # model1 for class1
            clf.fit(X_train,y_train1)
            print("The best parameters for model 1 are %s with a score of %0.2f"% (clf.best_params_, clf.best_score_))
            predicted_test1 = clf.predict_proba(X_test)
            predicted_train1 = clf.predict_proba(X_train)
            predicted_1 = clf.predict(X_test)
            roc_auc_test1 = roc_auc_score(y_test1, predicted_1)
            allTestROCAUC1.append(roc_auc_test1)
            print("Test ROC AUC for model1: ", roc_auc_test1)

            #model2 for class2
            clf.fit(X_train,y_train2),
            print("The best parameters gor model 2 are %s with a score of %0.2f"% (clf.best_params_, clf.best_score_))
            predicted_test2 = clf.predict_proba(X_test)
            predicted_train2 = clf.predict_proba(X_train)
            predicted_2 = clf.predict(X_test)
            roc_auc_test2 = roc_auc_score(y_test2, predicted_2)
            allTestROCAUC2.append(roc_auc_test2)
            print("Test ROC AUC for model2: ", roc_auc_test2)

            #model3 for class3
            clf.fit(X_train,y_train3)
            print("The best parameters for model 3 are %s with a score of %0.2f"% (clf.best_params_, clf.best_score_))
            predicted_test3 = clf.predict_proba(X_test)
            predicted_train3 = clf.predict_proba(X_train)
            predicted_3 = clf.predict(X_test)
            roc_auc_test3 = roc_auc_score(y_test3, predicted_3)
            allTestROCAUC3.append(roc_auc_test3)
            print("Test ROC AUC for model3: ", roc_auc_test3)

            #prediction for test data
            predicted_test = []
            for i in range(len(predicted_test1)):
                value = max(predicted_test1[i,1],predicted_test2[i,1],predicted_test3[i,1])
                if value == predicted_test1[i,1]:
                    predicted_test.append(1)
                elif value == predicted_test2[i,1]:
                    predicted_test.append(2)
                else:
                    predicted_test.append(3)

            #preddiction for train data
            predicted_train = []
            for i in range(len(predicted_train1)):
                value = max(predicted_train1[i,1],predicted_train2[i,1],predicted_train3[i,1])
                if value == predicted_train1[i,1]:
                    predicted_train.append(1)
                elif value == predicted_train2[i,1]:
                    predicted_train.append(2)
                else:
                    predicted_train.append(3)

            acc_train = accuracy_score(y_train, predicted_train)
            prec_train = precision_score(y_train, predicted_train,average='micro')
            recall_train = recall_score(y_train, predicted_train,average='micro')
            allTrainAccuracy.append(acc_train)
            allTrainPrec.append(prec_train)
            allTrainRecall.append(recall_train)
            print("Train Accuracy: ",acc_train)
            print("Train Precision: ",prec_train)
            print("Train Recall: ",recall_train)
            acc_test = accuracy_score(y_test, predicted_test)
            prec_test = precision_score(y_test, predicted_test,average='micro')
            recall_test = recall_score(y_test, predicted_test,average='micro')
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
        plt.plot([x for x in range(1, 11)], allTestROCAUC1)
        plt.title('ROC AUC for class 1 for all folds:' + kern)
        plt.xlabel('FOLDS')
        plt.ylabel('ROC AUC')
        plt.show()
        plt.plot([x for x in range(1, 11)], allTestROCAUC2)
        plt.title('ROC AUC for class 2 for all folds:' + kern)
        plt.xlabel('FOLDS')
        plt.ylabel('ROC AUC')
        plt.show()
        plt.plot([x for x in range(1, 11)], allTestROCAUC3)
        plt.title('ROC AUC for class 3 for all folds:' + kern)
        plt.xlabel('FOLDS')
        plt.ylabel('ROC AUC')
        plt.show()

if __name__ == '__main__':
    main()

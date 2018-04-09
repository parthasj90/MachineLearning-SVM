import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score , make_scorer,roc_auc_score,roc_curve,auc
from sklearn.metrics import precision_score
def main():
    filename = 'optdigits.tra'
    dataset = pd.read_csv(filename,header=None)
    X_train = dataset.iloc[:,:-1].values
    y_train = dataset.iloc[:,-1].values

    testfilename = 'optdigits.tes'
    dataset_test = pd.read_csv(testfilename,header=None)
    X_test = dataset_test.iloc[:,:-1].values
    y_test = dataset_test.iloc[:,-1].values

    #from sklearn.utils import shuffle
    #X,y = shuffle(X,y,random_state=13)
    y0 = np.array([1 if n == 0 else 0 for n in y_train])
    y1 = np.array([1 if n == 1 else 0 for n in y_train])
    y2 = np.array([1 if n == 2 else 0 for n in y_train])
    y3 = np.array([1 if n == 3 else 0 for n in y_train])
    y4 = np.array([1 if n == 4 else 0 for n in y_train])
    y5 = np.array([1 if n == 5 else 0 for n in y_train])
    y6 = np.array([1 if n == 6 else 0 for n in y_train])
    y7 = np.array([1 if n == 7 else 0 for n in y_train])
    y8 = np.array([1 if n == 8 else 0 for n in y_train])
    y9 = np.array([1 if n == 9 else 0 for n in y_train])

    from sklearn.preprocessing import StandardScaler

    kernels  = ['rbf','linear']
    for kern in kernels:
        print("Executing kernel: ",kern)

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
        helper(X_train,X_test,clf,y0)
        helper(X_train, X_test, clf, y1)
        helper(X_train, X_test, clf, y2)
        helper(X_train, X_test, clf, y3)
        helper(X_train, X_test, clf, y4)
        helper(X_train, X_test, clf, y5)
        helper(X_train, X_test, clf, y6)
        helper(X_train, X_test, clf, y7)
        helper(X_train, X_test, clf, y8)
        helper(X_train, X_test, clf, y9)

def helper(X_train,X_test,clf,y0,y_test):
    clf.fit(X_train,y0)
    print("The best parameters for model 1 are %s with a score of %0.2f"% (clf.best_params_, clf.best_score_))
    #predicted_test1 = clf.predict_proba(X_test)
    #predicted_train1 = clf.predict_proba(X_train)
    predicted_0 = clf.predict(X_train)
    acc_train = accuracy_score(y0, predicted_0)
    prec_train = precision_score(y0, predicted_0,average='micro')
    recall_train = recall_score(y0, predicted_0,average='micro')
    print("Train Accuracy: ",acc_train)
    print("Train Precision: ",prec_train)
    print("Train Recall: ",recall_train)
    predicted_1 = clf.predict(X_test)
    acc_train = accuracy_score(y_test, predicted_1)
    prec_train = precision_score(y_test, predicted_1,average='micro')
    recall_train = recall_score(y_test, predicted_1,average='micro')
    print("Train Accuracy: ",acc_train)
    print("Train Precision: ",prec_train)
    print("Train Recall: ",recall_train)
    # calculate the fpr and tpr for all thresholds of the classification
    probs = clf.predict_proba(X_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
if __name__ == '__main__':
    main()

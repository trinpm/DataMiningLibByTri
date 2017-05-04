import numpy as np
import scipy.stats as stats
import math

from numpy import unravel_index

from sklearn import model_selection
from sklearn.metrics import accuracy_score

# import models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


def getACC(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    F1, sens, spec = 0, 0, 0
    MCC = 0
    pred_ACC = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            TP += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FP += 1
        if y_true[i] == y_pred[i] == 0:
            TN += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FN += 1

    if (TP + FN) != 0 and (TN + FP) != 0 and (TP + FP) != 0 and (TN + FN) != 0:
        F1 = float(2 * TP) / float(2 * TP + FP + FN)
        sens = float(TP) / float(TP + FN)
        spec = float(TN) / float(TN + FP)
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP)*(TP + FN) * (TN + FP)*(TN + FN))
        pred_ACC = float(TP + TN) / float(TP + TN + FP + FN)
        print "*** TP: {0}, FP:{1}, TN:{2}, FN:{3}".format(TP, FP, TN, FN)
    else:
        print "F1_err"
        # print "** F1 measure is n/a"
        # print "y_pred", y_pred
        # print "*** TP: {0}, FP:{1}, TN:{2}, FN:{3}".format(TP, FP, TN, FN)
        # print "** stat for y_true:\n", stats.itemfreq(y_true)
        # print "** stat for y_pred:\n", stats.itemfreq(y_pred)

    acc = accuracy_score(y_true, y_pred)

    return (F1, sens, spec, acc, MCC, pred_ACC)


def runClassifier(X, Y, k, model, svm_regu=1, svm_gamma=1, svm_degree=1, rf_ntrees=1, rf_leafsamples=1, nn_unit=1,
                  nn_function=1, nn_solver='adam'):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import precision_recall_fscore_support
    kf = KFold(n_splits=k)
    i = 1
    F1_arr = np.array([])
    sens_arr = np.array([])
    spec_arr = np.array([])
    acc_arr = np.array([])
    mcc_arr = np.array([])
    pred_acc_arr = np.array([])

    for train, test in kf.split(X):
        # print "fold: ", i
        # print("train_idx: %s; test_idx: %s" % (train, test))
        # print 'X_train: {0}\nY_train: {1}'.format(X[train], Y[train])
        # print 'X_test: {0}\nY_test: {1}'.format(X[test], Y[test])

        X_train = X[train]
        X_test = X[test]
        Y_train = Y[train]
        Y_test = Y[test]

        # model
        if model == "LogReg":
            Y_pred = runLogisticRegression(X_train, Y_train, X_test, svm_regu)

        elif model == "SVM-1":
            Y_pred = runSVM_lin_regu(X_train, Y_train, X_test, svm_regu)

        elif model == "SVM-2":
            Y_pred = runSVM_poly_gam(X_train, Y_train, X_test, svm_regu, svm_degree)

        elif model == "SVM-3":
            Y_pred = runSVM_rbf_gam(X_train, Y_train, X_test, svm_regu, svm_gamma)

        elif model == "DT":
            Y_pred = runDecisionTree(X_train, Y_train, X_test)

        elif model == "NB":
            Y_pred = runNaiveBayes(X_train, Y_train, X_test)

        elif model == "KNN":
            Y_pred = runKNN(X_train, Y_train, X_test)

        elif model == "LDA":
            Y_pred = runLDA(X_train, Y_train, X_test)

        elif model == "RF":
            Y_pred = runRandomForest(X_train, Y_train, X_test, rf_ntrees, rf_leafsamples)

        elif model == "NeuroNet":
            Y_pred = runNeuralNetwork(X_train, Y_train, X_test, nn_unit, nn_solver, nn_function)

        else:
            print 'Wrong selected model! ', model
            break

        # measure accuracy
        (F1, sens, spec, acc, mcc, pred_acc) = getACC(Y_test, Y_pred)
        F1_arr = np.append(F1_arr, F1)
        sens_arr = np.append(sens_arr, sens)
        spec_arr = np.append(spec_arr, spec)
        acc_arr = np.append(acc_arr, acc)
        mcc_arr = np.append(mcc_arr, mcc)
        pred_acc_arr = np.append(pred_acc_arr, pred_acc)

        i = i + 1

    F1_mean = F1_arr.mean()
    sens_mean = sens_arr.mean()
    spec_mean = spec_arr.mean()
    acc_mean = acc_arr.mean()
    mcc_mean = mcc_arr.mean()
    pred_acc_mean = pred_acc_arr.mean()

    print "F1: ", F1_mean
    print "Sensi.: ", sens_mean
    print "Spec.: ", spec_mean
    print "Acc.: ", acc_mean
    print "Mcc.:", mcc_mean
    print "pred. Acc.: ", pred_acc_mean

    return F1_mean, sens_mean, spec_mean, acc_mean, mcc_mean, pred_acc_mean


def runLogisticRegression(X_tr, Y_tr, X_test, regu):
    model = LogisticRegression(C=regu)
    model.fit(X_tr, Y_tr)
    # print "coefs: ", model.coef_
    Y_pred = model.predict(X_test)
    return Y_pred

def logisticRegression(X, Y, kfold, regu):
    model = LogisticRegression(C=regu)
    model.fit(X, Y)
    [ACC, F1] = getMeasure(model, X, Y, kfold)
    print "ACC = {}, F1 = {}".format(ACC, F1)
    # coefs = model.coef_
    # return coefs.flatten()
    return model

def runSVM(X_tr, Y_tr, X_test):
    model = SVC(kernel='linear')
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_test)
    return Y_pred

def runSVM_lin_regu(X_tr, Y_tr, X_test, regu):
    model = SVC(kernel='linear', C=regu)
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_test)
    return Y_pred

def SVM_lin_regu(X, Y, kfold, regu):
    model = SVC(kernel='linear', C=regu)
    model.fit(X, Y)
    [ACC, F1] = getMeasure(model, X, Y, kfold)
    print "ACC = {}, F1 = {}".format(ACC, F1)
    coefs = model.coef_
    return coefs.flatten()

def getMeasure(model, X, Y, kfold):
    ACC = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    F1 = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='f1')
    return [ACC.mean(), F1.mean()]

def runSVM_poly_gam(X_tr, Y_tr, X_test, regu, deg):
    model = SVC(kernel='poly', C=regu, degree=deg)
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_test)
    return Y_pred

def SVM_poly_gam(X, Y, kfold, regu, deg):
    model = SVC(kernel='poly', C=regu, degree=deg)
    model.fit(X, Y)
    [ACC, F1] = getMeasure(model, X, Y, kfold)
    print "ACC = {}, F1 = {}".format(ACC, F1)
    coefs = model.coef_
    return coefs.flatten()

def runSVM_rbf_gam(X_tr, Y_tr, X_test, regu, gam):
    model = SVC(kernel='rbf', C=regu, gamma=gam)
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_test)
    return Y_pred


def testLogisticRegression(X, Y, kfold, regu):
    # LOGISTIC REGRESSION
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=regu)
    mdl = model.fit(X,Y)
    # logReg = LogisticRegression(C=1e5)
    model.fit(X, Y)
    print 'model.coef_: ', model.coef_
    results_acc = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    results_f1 = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='f1')  # f1 for binary target
    print '1. ACC (Logistic Regression): ', results_acc.mean()
    print '1. F1 (Logistic Regression): ', results_f1.mean()


def testSVM(X, Y, kfold):
    # SVM
    model = SVC()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    results_f1 = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='f1')
    print '2. ACC (SVM): ', results.mean()
    print '2. F1 (SVM): ', results_f1.mean()


def testDecisionTree(X, Y, kfold):
    # DECISION TREE
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    print '3. ACC (Decision Tree): ', results.mean()


def runDecisionTree(X_tr, Y_tr, X_test):
    model = DecisionTreeClassifier()
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_test)
    return Y_pred


def testNaiveBayes(X, Y, kfold):
    # NAIVE BAYES
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    print '4. ACC (Naive Bayes): ', results.mean()


def runNaiveBayes(X_tr, Y_tr, X_test):
    model = GaussianNB()
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_test)
    return Y_pred

def testKNN(X, Y, kfold):
    model = KNeighborsClassifier()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    print '5. ACC (KNN): ', results.mean()

def runKNN(X_tr, Y_tr, X_test):
    model = KNeighborsClassifier()
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_test)
    return Y_pred

def testLDA(X, Y, kfold):
    # LINEAR DISCRIMINANT ANALYSIS
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model = LinearDiscriminantAnalysis()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    print '6. ACC (LDA): ', results.mean()

def runLDA(X_tr, Y_tr, X_test):
    model = LinearDiscriminantAnalysis()
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_test)
    return Y_pred

def testRandomForest(X, Y, kfold):
    # RANDOM FOREST
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    print '7. ACC (RandomForest): ', results.mean()

def createRandomForest(X, Y, mdlPath, ntrees, leafsamples):
    model = RandomForestClassifier(max_features="sqrt", n_estimators=ntrees, min_samples_leaf=leafsamples, oob_score=True)
    model.fit(X, Y)
    print "oob_score:", model.oob_score_
    #store trained model
    from sklearn.externals import joblib
    joblib.dump(model, mdlPath)

def runRandomForest(X_tr, Y_tr, X_test, ntrees, leafsamples):
    print "ntrees: ", ntrees
    model = RandomForestClassifier(max_features="sqrt", n_estimators=ntrees, min_samples_leaf=leafsamples, oob_score=True)
    model.fit(X_tr, Y_tr)
    print "oob_score:", model.oob_score_

    # print "shape of X_test: ", np.shape(X_test)
    # X_test = np.array([[1,2,3,4,5,6,7,8], [1,1,1,0,0,0,0,0]])
    # print "X_test: ", X_test
    # print "shape of X_test: ", np.shape(X_test)
    Y_pred = model.predict(X_test)
    # print "Y_pred: ", Y_pred
    return Y_pred


def testAdaBoost(X, Y, kfold):
    # ADA BOOST
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1.5,
                               algorithm="SAMME")
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    print '8. ACC (AdaBoost): ', results.mean()


def runAdaBoost(X_tr, Y_tr, X_test):
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1.5,
                               algorithm="SAMME")
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_test)
    return Y_pred

def runNeuralNetwork(X_tr, Y_tr, X_test, n_units, opt_weigh, func):
    # print "unit={}, alpha={}".format(n_units, _alpha)
    model = MLPClassifier(hidden_layer_sizes=(n_units, n_units, n_units, 2), solver=opt_weigh, alpha=0.00001,
                          activation=func)
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_test)
    return Y_pred



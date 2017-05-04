import numpy as np
from numpy import unravel_index
import ClassifyLib as cl


def tuneParametersForLogReg(X, Y, k):
    print "\n\n############### CV FOR Logistic Regression ###############"
    C = [ 0.0001, 0.001, 0.1, 1, 10 ]
    gamma = None
    degree = None
    F1_arr = np.array([])
    sens_arr = np.array([])
    spec_arr = np.array([])
    acc_arr = np.array([])
    mcc_arr = np.array([])
    pred_acc_arr = np.array([])
    for c in C:
        print "Regulization C = {}".format(c)
        [F1, sens, spec, acc, mcc, pred_acc] = cl.runClassifier(X, Y, k, "LogReg", c, gamma, degree)
        F1_arr = np.append(F1_arr, F1)
        sens_arr = np.append(sens_arr, sens)
        spec_arr = np.append(spec_arr, spec)
        acc_arr = np.append(acc_arr, acc)
        mcc_arr = np.append(mcc_arr, mcc)
        pred_acc_arr = np.append(pred_acc_arr, pred_acc)

    print "\n\n############### CV FOR Logistic Regression - Best parameters ###############"
    print "C: ", C

    print "ACC.: \n", acc_arr
    idx = np.argmax(acc_arr)
    print "C: ", C[idx]
    print "Best ACC: ", acc_arr[idx]
    print "with F1: ", F1_arr[idx]
    print "with sens: ", sens_arr[idx]
    print "with spec: ", spec_arr[idx]
    print "with mcc: ", mcc_arr[idx]
    print "with pred. acc: ", pred_acc_arr[idx]

    print "F1: \n", F1_arr
    idx = np.argmax(F1_arr)
    print "C: ", C[idx]
    print "Best F1: ", F1_arr[idx]
    print "with ACC: ", acc_arr[idx]
    print "with sens: ", sens_arr[idx]
    print "with spec: ", spec_arr[idx]
    print "with mcc: ", mcc_arr[idx]
    print "with pred. acc: ", pred_acc_arr[idx]

    print "MCC: \n", mcc_arr
    idx = np.argmax(mcc_arr)
    print "C: ", C[idx]
    print "Best MCC: ", mcc_arr[idx]
    print "with F1: ", F1_arr[idx]
    print "with ACC: ", acc_arr[idx]
    print "with sens: ", sens_arr[idx]
    print "with spec: ", spec_arr[idx]
    print "with pred. acc: ", pred_acc_arr[idx]

def maxACC(C, acc_arr, F1_arr, sens_arr, spec_arr, mcc_arr):
    # sort by ACC
    print "ACC.: \n", acc_arr
    idx = np.argmax(acc_arr)
    print "Best ACC: ", acc_arr[idx]
    print "with C: ", C[idx]
    print "with F1: ", F1_arr[idx]
    print "with sens: ", sens_arr[idx]
    print "with spec: ", spec_arr[idx]
    print "with mcc: ", mcc_arr[idx]

def maxF1(C, F1_arr, acc_arr, sens_arr, spec_arr, mcc_arr):
    # sort by F1
    print "F1.: \n", F1_arr
    idx = np.argmax(F1_arr)
    print "Best F1: ", F1_arr[idx]
    print "with C: ", C[idx]
    print "with ACC: ", acc_arr[idx]
    print "with sens: ", sens_arr[idx]
    print "with spec: ", spec_arr[idx]
    print "with mcc: ", mcc_arr[idx]

def maxMCC(C, F1_arr, acc_arr, sens_arr, spec_arr, mcc_arr):
    # sort by mcc
    print "MCC.: \n", mcc_arr
    idx = np.argmax(mcc_arr)
    print "Best MCC: ", mcc_arr[idx]
    print "with C: ", C[idx]
    print "with F1: ", F1_arr[idx]
    print "with ACC: ", acc_arr[idx]
    print "with sens: ", sens_arr[idx]
    print "with spec: ", spec_arr[idx]

def maxACC_2(C, acc_m, X, f1_m, sens_m, spec_m, mcc_m):
    # sort by ACC
    print "ACC. :\n", acc_m
    (x, y) = unravel_index(acc_m.argmax(), acc_m.shape)
    print "x = {}, y = {}".format(x, y)
    print "Best ACC: ", acc_m[x, y]
    print "with C = {}, with X = {}".format(C[x], X[y])
    print "with F1: ", f1_m[x, y]
    print "with sens: ", sens_m[x, y]
    print "with spec: ", spec_m[x, y]
    print "with mcc: ", mcc_m[x, y]

def maxF1_2(C, acc_m, X, f1_m, sens_m, spec_m, mcc_m):
    # sort by F1
    print "F1. :\n", f1_m
    (x, y) = unravel_index(f1_m.argmax(), f1_m.shape)
    print "x = {}, y = {}".format(x, y)
    print "Best F1: ", f1_m[x, y]
    print "with C = {}, with X = {}".format(C[x], X[y])
    print "with ACC: ", acc_m[x, y]
    print "with sens: ", sens_m[x, y]
    print "with spec: ", spec_m[x, y]
    print "with mcc: ", mcc_m[x, y]


def tuneParametersForSVM(X, Y, k):
    print "\n\n############### CV FOR SVM Linear Kernel ###############"
    C = [0.0001, 0.001, 0.1, 1, 10]
    gamma = None
    degree = None
    F1_arr = np.array([])
    sens_arr = np.array([])
    spec_arr = np.array([])
    acc_arr = np.array([])
    mcc_arr = np.array([])
    for c in C:
        print "Regulization C = {}".format(c)
        print "X: ", X
        print "Y: ", Y
        print "X.size: {}, Y.size: {}".format(len(X), len(Y))
        [F1, sens, spec, acc, mcc] = cl.runClassifier(X, Y, k, "SVM-1", c, gamma, degree)
        F1_arr = np.append(F1_arr, F1)
        sens_arr = np.append(sens_arr, sens)
        spec_arr = np.append(spec_arr, spec)
        acc_arr = np.append(acc_arr, acc)
        mcc_arr = np.append(mcc_arr, mcc)

    print "\n\n############### CV FOR SVM Linear Kernel ###############"
    print "C: ", C
    maxACC(C, acc_arr, F1_arr, sens_arr, spec_arr, mcc_arr)
    maxF1(C, acc_arr, F1_arr, sens_arr, spec_arr, mcc_arr)
    maxMCC(C, acc_arr, F1_arr, sens_arr, spec_arr, mcc_arr)


    print "\n\n############### CV FOR SVM Poly Kernel ###############"
    C = [0.0001, 0.001, 0.1, 1, 10]
    gamma = None
    degree = [1, 2, 4, 8]
    f1_m = np.empty((len(C), len(degree)))
    sens_m = np.empty((len(C), len(degree)))
    spec_m = np.empty((len(C), len(degree)))
    acc_m = np.empty((len(C), len(degree)))
    mcc_m = np.empty((len(C), len(degree)))
    for i in range(0, len(C)):
        c = C[i]
        for j in range(0, len(degree)):
            d = degree[j]
            print "Regulization C = {}, Degree = {}".format(c, d)
            print "X: ", X
            print "Y: ", Y
            print "X.size: {}, Y.size: {}".format(len(X), len(Y))
            [f1, sens, spec, acc, mcc] = cl.runClassifier(X, Y, k, "SVM-2", c, gamma, d)
            f1_m[i][j] = f1
            sens_m[i][j] = sens
            spec_m[i][j] = spec
            acc_m[i][j] = acc
            mcc_m[i][j] = mcc
            # print "Row = {}, Col = {}, err = {}".format(i, j, err[i][j])
    print "\n\n############### CV FOR SVM Poly Kernel ###############"
    print "C: ", C
    print "degree: ", degree
    print "MCC: \n", mcc_m
    metric = degree
    maxACC_2(C, acc_m, metric, f1_m, sens_m, spec_m, mcc_m)
    maxF1_2(C, acc_m, metric, f1_m, sens_m, spec_m, mcc_m)


    print "\n\n############### CV FOR SVM RBF Kernel ###############"
    C = [0.0001, 0.001, 0.1, 1, 10]
    gamma = [0.05, 0.5, 5, 10]
    degree = None
    f1_m = np.empty((len(C), len(gamma)))
    sens_m = np.empty((len(C), len(gamma)))
    spec_m = np.empty((len(C), len(gamma)))
    acc_m = np.empty((len(C), len(gamma)))
    mcc_m = np.empty((len(C), len(gamma)))
    for i in range(0, len(C)):
        c = C[i]
        for j in range(0, len(gamma)):
            g = gamma[j]
            print "Regulization C = {}, Gamma = {}".format(c, g)
            print "X: ", X
            print "Y: ", Y
            print "X.size: {}, Y.size: {}".format(len(X), len(Y))
            [f1, sens, spec, acc, mcc] = cl.runClassifier(X, Y, k, "SVM-3", c, g, degree)
            f1_m[i][j] = f1
            sens_m[i][j] = sens
            spec_m[i][j] = spec
            acc_m[i][j] = acc
            mcc_m[i][j] = mcc
            # print "sens_m = {}, sens = {}".format(sens_m[i][j], sens)
    print "\n\n############### CV FOR SVM RBF Kernel ###############"
    print "C: ", C
    print "gamma: ", gamma
    print "MCC: ", mcc_m
    metric = gamma
    maxACC_2(C, acc_m, metric, f1_m, sens_m, spec_m, mcc_m)
    maxF1_2(C, acc_m, metric, f1_m, sens_m, spec_m, mcc_m)


def tuneParametersForRandomForest(X, Y, k):
    print "\n\n############### CV FOR Random Forest ###############"

    n_estimators = [1, 10, 20, 30]
    leaf_sizes = [1, 5, 10, 50, 100, 200, 500]

    C = None
    gamma = None
    degree = None
    f1_m = np.empty((len(n_estimators), len(leaf_sizes)))
    sens_m = np.empty((len(n_estimators), len(leaf_sizes)))
    spec_m = np.empty((len(n_estimators), len(leaf_sizes)))
    acc_m = np.empty((len(n_estimators), len(leaf_sizes)))

    for i in range(0, len(n_estimators)):
        n_trees = n_estimators[i]
        for j in range(0, len(leaf_sizes)):
            s_leaf = leaf_sizes[j]
            [f1, sens, spec, acc] = cl.runClassifier(X, Y, k, "RF", svm_regu=None, svm_gamma=None, svm_degree=None,
                                                     rf_ntrees=n_trees, rf_leafsamples=s_leaf)
            f1_m[i][j] = f1
            sens_m[i][j] = sens
            spec_m[i][j] = spec
            acc_m[i][j] = acc
            # print "sens_m = {}, sens = {}".format(sens_m[i][j], sens)
    print "\n\n############### CV FOR Random Forest ###############"
    print "n_trees: ", n_estimators
    print "leaf_sizes: ", leaf_sizes
    print "ACC. :\n", acc_m
    (x, y) = unravel_index(acc_m.argmax(), acc_m.shape)
    print "x = {}, y = {}".format(x, y)
    print "Best n_tree = {}, Best s_leaf = {}".format(n_estimators[x], leaf_sizes[y])
    print "Best ACC: ", acc_m[x, y]
    print "with F1: ", f1_m[x, y]
    print "with sens: ", sens_m[x, y]
    print "with spec: ", spec_m[x, y]

    print "F1. :\n", f1_m
    (x, y) = unravel_index(f1_m.argmax(), f1_m.shape)
    print "x = {}, y = {}".format(x, y)
    print "Best n_tree = {}, Best s_leaf = {}".format(n_estimators[x], leaf_sizes[y])
    print "Best F1: ", f1_m[x, y]
    print "with ACC: ", acc_m[x, y]
    print "with sens: ", sens_m[x, y]
    print "with spec: ", spec_m[x, y]


def tuneParametersForNeuralNetwork(X, Y, k):
    print "\n\n############### CV FOR Neuro Network ###############"

    solvers = ['lbfgs', 'sgd', 'adam']
    n_units = [100, 200, 300]
    activations = ['identity', 'logistic', 'tanh', 'relu']

    C = None
    gamma = None
    degree = None
    for solver in solvers:
        f1_m = np.empty((len(n_units), len(activations)))
        sens_m = np.empty((len(n_units), len(activations)))
        spec_m = np.empty((len(n_units), len(activations)))
        acc_m = np.empty((len(n_units), len(activations)))

        for i in range(0, len(n_units)):
            unit = n_units[i]
            for j in range(0, len(activations)):
                act_func = activations[j]
                [f1, sens, spec, acc] = cl.runClassifier(X, Y, k, "NeuroNet", svm_regu=None, svm_gamma=None, svm_degree=None,
                                                         rf_ntrees=None, rf_leafsamples=None, nn_unit=unit, nn_function=act_func, nn_solver=solver)
                f1_m[i][j] = f1
                sens_m[i][j] = sens
                spec_m[i][j] = spec
                acc_m[i][j] = acc
                # print "sens_m = {}, sens = {}".format(sens_m[i][j], sens)
        print "\n\n############### CV FOR Neuro Network ###############"
        print "solver (func. to opt. weight): ", solver
        print "n_unit: ", n_units
        print "act_func: ", activations
        print "ACC. :\n", acc_m
        (x, y) = unravel_index(acc_m.argmax(), acc_m.shape)
        print "x = {}, y = {}".format(x, y)
        print "Best n_unit = {}, Best act_func = {}".format(n_units[x], activations[y])
        print "Best ACC: ", acc_m[x, y]
        print "with F1: ", f1_m[x, y]
        print "with sens: ", sens_m[x, y]
        print "with spec: ", spec_m[x, y]

        print "F1. :\n", f1_m
        (x, y) = unravel_index(f1_m.argmax(), f1_m.shape)
        print "x = {}, y = {}".format(x, y)
        print "Best n_unit = {}, Best act_func = {}".format(n_units[x], activations[y])
        print "Best F1: ", f1_m[x, y]
        print "with ACC: ", acc_m[x, y]
        print "with sens: ", sens_m[x, y]
        print "with spec: ", spec_m[x, y]
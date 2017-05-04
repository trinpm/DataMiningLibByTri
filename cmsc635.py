import Utils as ut
import ClassifyLib as cl
import numpy as np
import ParameterTuning as pt
import FeatureSelection as fs
import sys

def extractData(df):
    [row, col] = df.shape
    print df.shape

    data = df.values
    X = data[:, 0: col - 1]
    Y = data[:, col - 1]

    # 1.checking few data
    print df.head(5)
    print "first few X:\n", X[:5]
    print "last few X:\n", X[-5:]
    print "first few Y:\n", Y[:5]
    print "last few Y:\n", Y[-5:]

    return [X, Y]


def convertLabel(Y, label):
    # print "Y.shape: ", Y.shape
    Y_new = np.zeros(len(Y))
    for i in range(0, len(Y)):
        if Y[i] == label:
            Y_new[i] = 1
        else:
            Y_new[i] = 0
    return Y_new

def getBinaryLabel(Y):
    Y_class0 = convertLabel(Y, "zero")
    Y_class1 = convertLabel(Y, "one")
    Y_class2 = convertLabel(Y, "two")
    Y_class3 = convertLabel(Y, "three")

    print "Y_class0_head={}\nY_class0_tail={}".format(Y_class0[:5], Y_class0[-5:])
    print "Y_class1_head={}\nY_class1_tail={}".format(Y_class1[:5], Y_class1[-5:])
    print "Y_class2_head={}\nY_class2_tail={}".format(Y_class2[:5], Y_class2[-5:])
    print "Y_class3_head={}\nY_class3_tail={}".format(Y_class3[:5], Y_class3[-5:])

    return [Y_class0, Y_class1, Y_class2, Y_class3]

def parameterSelection(X, Y, kfold):
    pt.tuneParametersForLogReg(X, Y, kfold)

def getTextLabel(list_in):
    list_out = []
    cls0_cnt = 0
    cls1_cnt = 0
    cls2_cnt = 0
    cls3_cnt = 0
    for l in list_in:
        cls = ""
        if l == 0:
            cls = "zero"
            cls0_cnt = cls0_cnt + 1
        elif l == 1:
            cls = "one"
            cls1_cnt = cls1_cnt + 1
        elif l == 2:
            cls = "two"
            cls2_cnt = cls2_cnt + 1
        elif l == 3:
            cls = "three"
            cls3_cnt = cls3_cnt + 1
        list_out.append(cls)

    print "pred cls0: ", cls0_cnt
    print "pred cls1: ", cls1_cnt
    print "pred cls2: ", cls2_cnt
    print "pred cls3: ", cls3_cnt
    return list_out

def runTest(X_test, clsifier_0, clsifier_1, clsifier_2, clsifier_3):
    print "order of classes: ", clsifier_0.classes_

    print "type of X_test: ", type(X_test)
    pred_cls_all = []
    for x in X_test:
        x = x.reshape(1, -1)
        # print "x: ", x
        prob = []
        p0 = clsifier_0.predict_proba(x)
        prob.append(p0[:,1])
        p1 = clsifier_1.predict_proba(x)
        prob.append(p1[:,1])
        p2 = clsifier_2.predict_proba(x)
        prob.append(p2[:,1])
        p3 = clsifier_3.predict_proba(x)
        prob.append(p3[:,1])

        # print "p0:{}\np1:{}\np2:{}\np3:{}".format(p0, p1, p2, p3)

        max_pred_prob = max(prob)
        pred_cls = prob.index(max_pred_prob)
        # print "max probability:{} -> class:{}".format(max_pred_prob, pred_cls)
        pred_cls_all.append(pred_cls)

    return getTextLabel(pred_cls_all)

def getACC(Y_pred, Y_true):
    if(len(Y_pred) == len(Y_true)):
        cnt = 0
        for i in range(0, len(Y_pred)):
            if Y_pred[i] == Y_true[i]:
                cnt = cnt + 1
    print "#correctly predicted: ", cnt
    return float(float(cnt)/float(len(Y_pred)))

def doNorm(X):
    X = ut.minMaxScale(X)
    return X

def doFeatureSel(X, Y):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif, mutual_info_classif
    kbest = SelectKBest(score_func=mutual_info_classif, k=600)  # {150, 300, 600}
    X_new = kbest.fit_transform(X, Y)
    sel_features = kbest.get_support(indices=True)
    print "type(features): ", type(sel_features)
    print "features: ", sel_features

    print "X_new first few:\n", X_new[:5]
    print "X_new last few:\n", X_new[-5:]

    X_sel = X[:,sel_features]
    print "X_sel first few:\n", X_sel[:5]
    print "X_sel last few:\n", X_sel[-5:]

    print 'X.shape (top-k) features: ', X_sel.shape
    return [X_sel, sel_features]

def runClassification(X, Y, normFlag, X_test, Y_test, url_test):

    feature_selection = 1
    model_selection = 1
    train =0
    test = 0
    kfold = 5

    #normalize data
    if normFlag:
        X = doNorm(X)
        print "first few X (normed):\n", X[:5]
        print "last few X (normed):\n", X[-5:]
        print "first few Y (normed):\n", Y[:5]
        print "last few Y (normed):\n", Y[-5:]

    # select important features ????

    [row, col] = X.shape
    print 'X.shape: ', X.shape

    if feature_selection:
        [X, sel_features] = doFeatureSel(X, Y)

    # 1. get 1 vs. all labels:
    [Y_cls0, Y_cls1, Y_cls2, Y_cls3] = getBinaryLabel(Y)
    X_cls0 = X
    X_cls1 = X
    X_cls2 = X
    X_cls3 = X

    if model_selection:
        # 2. model selection:
        print "selecting parameter for class 0..."
        # [X_cls0, Y_cls0] = ut.balanceData(X_cls0, Y_cls0, 0.9) #no need
        parameterSelection(X_cls0, Y_cls0, kfold)

        print "selecting parameter for class 1..."
        [X_cls1, Y_cls1] = ut.balanceData(X_cls1, Y_cls1, 0.7) #0.7
        parameterSelection(X_cls1, Y_cls1, kfold)

        print "selecting parameter for class 2..."
        [X_cls2, Y_cls2] = ut.balanceData(X_cls2, Y_cls2, 0.7) #0.65
        parameterSelection(X_cls2, Y_cls2, kfold)

        print "selecting parameter for class 3..."
        [X_cls3, Y_cls3] = ut.balanceData(X_cls3, Y_cls3, 0.7) #no need
        parameterSelection(X_cls3, Y_cls3, kfold)

    clsifier_0 = None
    clsifier_1 = None
    clsifier_2 = None
    clsifier_3 = None
    if train:
        # 3. build 4 classifiers
        print "building classifiers..."

        # 3.1 build classifier-0 to classify "zero" vs. other
        clsifier_0 = cl.logisticRegression(X_cls0, Y_cls0, kfold, 0.1)

        # 3.2 build classifier-1 to classify "one" vs. other
        [X_cls1, Y_cls1] = ut.balanceData(X_cls1, Y_cls1, 0.7)
        clsifier_1 = cl.logisticRegression(X_cls1, Y_cls1, kfold, 10)

        # 3.3 build classifier-2 to classify "two" vs. other
        [X_cls2, Y_cls2] = ut.balanceData(X_cls2, Y_cls2, 0.7)
        clsifier_2 = cl.logisticRegression(X_cls2, Y_cls2, kfold, 10)

        # 3.4 build classifier-3 to classify "three" vs. other
        [X_cls3, Y_cls3] = ut.balanceData(X_cls3, Y_cls3, 0.7)
        clsifier_3 = cl.logisticRegression(X_cls3, Y_cls3, kfold, 0.1)

    if test:
        ############# TESTING #############

        if normFlag:
            X_test = doNorm(X_test)
            print "first few X_test (normed):\n", X[:5]
            print "last few X_test (normed):\n", X[-5:]

        if feature_selection:
            print "X_test (first few): ", X_test[:,(0,1,2,3,4)]
            X_test = X_test[:,sel_features]
            print "shape(X_test): ", X_test.shape
            print "sel_features: ", sel_features
            print "X_test (first few): ", X_test[:,(0,1,2,3,4)]

        # get some X to test:
        # X_test = X
        # Y_test = Y
        # print "X_test: ", X_test
        # print "Y_test: ", Y_test
        # print "X_test.shape: ", X_test.shape

        Y_pred = runTest(X_test, clsifier_0, clsifier_1, clsifier_2, clsifier_3)
        print "predicted labels: ", Y_pred
        ut.writeList2file(url_test + ".out", Y_pred)

        # acc = getACC(Y_pred, Y_test)
        # print "pct. of correctly predicted: ", acc*100, "%"
        ####################################

def main():
    #load data
    url = "/home/student/trinpm/classes/spring17/CMSC635/project/training.csv"
    # url = "/home/student/trinpm/classes/spring17/CMSC635/project/training_reduced.csv"
    # url = "/home/student/trinpm/classes/spring17/CMSC635/project/training_1000.csv"

    df = ut.loadCSV(url, ',')
    [X, Y] = extractData(df)

    #get test data:
    url_test = "/home/student/trinpm/classes/spring17/CMSC635/project/test.csv"
    df_test = ut.loadCSV(url_test, ',')
    [X_test, Y_test] = extractData(df_test)

    normFlag = True
    for i in range(0, 1):
        runClassification(X, Y, normFlag, X_test, Y_test, url_test + "." + str(i))

if __name__ == '__main__':
    main()
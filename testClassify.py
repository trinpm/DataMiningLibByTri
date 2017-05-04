import pandas
import numpy as np
from sklearn import model_selection
import sys
import Utils as ut

# User-defined functions
import ClassifyLib as cl
import ParameterTuning as pt
import FeatureSelection as fs
import RegressLib as rl

def loadCSV(url, names):
    df = pandas.read_csv(url, names=names)
    return df


def testClassifers(X, Y, kfold):
    print ("-------------- TEST CLASSIFIERS --------------")
    cl.testLogisticRegression(X, Y, kfold)
    cl.testSVM(X, Y, kfold)
    cl.testDecisionTree(X, Y, kfold)
    cl.testNaiveBayes(X, Y, kfold)
    cl.testKNN(X, Y, kfold)
    cl.testLDA(X, Y, kfold)
    cl.testRandomForest(X, Y, kfold)
    # cl.testAdaBoost(X, Y, kfold)
    print ("----------------------------------------------")


def runClassifiers(X, Y, k):
    print ("-------------- RUN CLASSIFIERS --------------")
    print ("--------------")
    print "Logistic Regression"
    cl.runClassifier(X, Y, k, "LogReg")
    print ("--------------")
    print 'SVM_lin_regu'
    cl.runClassifier(X, Y, k, "SVM-1")
    print ("--------------")
    print 'SVM_lin_gam'
    cl.runClassifier(X, Y, k, "SVM-2")
    print ("--------------")
    print 'SVM_poly_gam'
    cl.runClassifier(X, Y, k, "SVM-3")
    print ("--------------")
    print 'Decision Tree'
    cl.runClassifier(X, Y, k, "DT")
    print ("--------------")
    print 'Naive Bayes'
    cl.runClassifier(X, Y, k, "NB")
    print ("--------------")
    print 'K-Nearest Neighbors'
    cl.runClassifier(X, Y, k, "KNN")
    print ("--------------")
    print 'Linear Discriminant Analysis'
    cl.runClassifier(X, Y, k, "LDA")
    print ("--------------")
    print 'Random Forest'
    cl.runClassifier(X, Y, k, "RF")
    print ("--------------")
    print ("----------------------------------------------")


def getKfolds(split):
    seed = 7
    kfolds = model_selection.KFold(n_splits=split, random_state=seed)
    return kfolds


def testDiabeteData():
    # TEST WITH DIABETES DATA FROM UCI
    url = "pima-indians-diabetes.data"
    print url
    headers = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    df = loadCSV(url, headers)

    # df = df.iloc[::2]

    print "First few samples: \n", df.head(5)
    # scale data
    df[['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']] = ut.minMaxScale(
        df[['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']])
    # df = df[['preg', 'plas', 'pres', 'mass', 'pedi', 'age', 'class']]

    array = df.values
    [nrow, ncol] = array.shape
    print "First few samples: \n", df.head(5)
    # print "Dimension: ", array.shape

    X = array[:, 0:(ncol - 1)]
    Y = array[:, (ncol - 1)]

    # kfold = getKfolds(5)
    # testClassifers(X, Y, kfold)
    # k = 5
    # runClassifiers(X, Y, k)

    # Feature Selection
    m = ncol - 1  # Number of features
    print "\nFeature Selection by RFE (smaller is better): "
    fs.recursiveFeatureElimination(X, Y, m)

    print "\nFeature Selection by kBest (higher values mean higher dependency): "
    fs.selectKBest(X, Y, m)

    print "\nFeature Selection by extraTree (higher is better): "
    fs.extraTrees(X, Y)

    # Tuning Parameters
    # pt.tuneParametersForLogReg(X, Y, k)
    # pt.tuneParametersForSVM(X, Y, k)
    # pt.tuneParametersForNeuralNetwork(X, Y, k)

def testNetworkProbing(url, k, mdlDumpPath):

    norm = 0
    visualize = 0
    tuning = 0
    training = 1

    print url
    print mdlDumpPath
    headers = ['redn_s', 'yeln_s', 'redn_odeg', 'ttl_RR_edges', 'm', 'k', 'ttl_links2_green', 'm_prime',
               'ttl_links2_yeln', 'u_2_yeln', 'odeg', 'bc', 'cc', 'eig', 'pr', 'clc', 'glb_indeg', 'glb_odeg', 'glb_bc', 'glb_cc', 'glb_eig', 'glb_pr',
               'glb_clc', 'deg2hops', 'cosin_simi', 'euclid_simi']

    df = loadCSV(url, headers)

    # select portions of data
    #dfSel = df[['redn_s', 'yeln_s', 'redn_odeg', 'ttl_RR_edges', 'm', 'k', 'odeg', 'cc', 'eig', 'pr', 'clc', 'deg2hops', 'cosin_simi', 'euclid_simi']]
    dfSel = df[[ 'redn_s', 'yeln_s', 'redn_odeg', 'ttl_RR_edges', 'm', 'k', 'odeg', 'eig', 'pr', 'clc' ]]

    print "First few samples: \n", dfSel.head(5)
    print "Describe: \n ", dfSel.describe()

    if (norm):
        X = ut.minMaxScale(dfSel[['redn_s', 'yeln_s', 'redn_odeg', 'ttl_RR_edges', 'm', 'k', 'odeg', 'eig', 'pr', 'clc' ]])
        # print "First few samples: \n", dfSel.head(5)
        # print dfSel.describe()
    else:
        X = dfSel.values

    # print array
    [nrow, ncol] = X.shape
    print "Dimension: ", X.shape

    print "X: ", X[:3]
    Y = df[['ttl_links2_green']].values # get gain
    Y = Y.ravel()
    print "Y: ", Y[:3]

    # visualize data
    if visualize:
        import matplotlib.pyplot as plt
        dfSel.hist()
        plt.show()

    # tuning parameter
    if tuning:
        rl.tuneLassoRegression(X, Y)
        rl.tuneLinearSVR(X, Y)
        rl.tuneSVMRegressor(X, Y)

    # train
    if training:
        # mdl = rl.svmRegressor(X, Y)
        mdl = rl.linearRegression(X, Y)
        print "score: {}, coefs: {} ".format(mdl.score(X, Y), mdl.coef_)
        # mdl = rl.ridgeRegression(X, Y)
        # print "score: {}, coefs: {} ".format(mdl.score(X, Y), mdl.coef_)
        # mdl = rl.lassoRegression(X, Y)
        # print "score: {}, coefs: {} ".format(mdl.score(X, Y), mdl.coef_)
        # mdl = rl.linearSVR(X, Y)


        # test
        # X = np.array([[1, 0.0003, 0.198, 0.018, 0, 2.16, 25.7, 172.2]])
        # print "predict: ", mdl.predict(X)

        # dump model
        from sklearn.externals import joblib
        joblib.dump(mdl, mdlDumpPath)
        print "dumped model to ", mdlDumpPath

def testNetworkData(url, k, bflag, bratio):
    headers = ['redn_s', 'yeln_s', 'redn_odeg', 'ttl_RR_edges', 'm', 'k', 'ttl_links2_greenn', 'm_prime',
               'ttl_links2_yeln', 'u_2_yeln',
               'odeg', 'bc', 'cc', 'eig', 'pr', 'clc', 'glb_indeg', 'glb_odeg', 'glb_bc', 'glb_cc', 'glb_eig', 'glb_pr',
               'glb_clc', 'deg2hops', 'cosin_simi', 'euclid_simi']
    df = loadCSV(url, headers)

    # 0:normalNode/1:hairNode,
    df['class'] = df['glb_indeg'] == 1
    df['class'] = np.where(df['class'], 1, 0)

    # print "Count #samples in each class:"
    # print df['class'].value_counts()

    # select portions of data
    dfSel = df[['odeg', 'cc', 'eig', 'pr', 'clc', 'deg2hops', 'cosin_simi', 'euclid_simi', 'class']]
    print "First few samples: \n", dfSel.head(5)

    # scale
    dfSel[['odeg', 'cc', 'eig', 'pr', 'clc', 'deg2hops', 'cosin_simi', 'euclid_simi', 'class']] = ut.minMaxScale(
        dfSel[['odeg', 'cc', 'eig', 'pr', 'clc', 'deg2hops', 'cosin_simi', 'euclid_simi', 'class']])
    # print "First few samples: \n", dfSel.head(5)
    # print dfSel.describe()

    # after selecting features
    # dfSel = df[['eig', 'pr', 'deg2hops', 'class']]

    array = dfSel.values
    # print array
    [nrow, ncol] = array.shape  # index starts from 1
    print "Dimension: ", array.shape

    X = array[:, 0:(ncol - 1)]  # index starts from 0
    # print "X:", X
    Y = array[:, (ncol - 1)]  # label
    # print Y

    # kfold = getKfolds(k)
    # testClassifers(X, Y, kfold)

    if bflag == True:
        (X, Y) = ut.balanceData(X, Y, bratio)
    # runClassifiers(X, Y, k)

    # Feature Selection
    # m = ncol - 1  # Number of features
    # print "\nFeature Selection by RFE (smaller is better): "
    # fs.recursiveFeatureElimination(X, Y, m)
    # print "\nFeature Selection by kBest (higher values mean higher dependency): "
    # fs.selectKBest(X, Y, m)
    # print "\nFeature Selection by extraTree (higher is better): "
    # fs.extraTrees(X, Y)

    # Tuning Parameters
    #pt.tuneParametersForLogReg(X, Y, k)
    #pt.tuneParametersForSVM(X, Y, k)
    # pt.tuneParametersForRandomForest(X, Y, k)
    #pt.tuneParametersForNeuralNetwork(X, Y, k)

    # Run all classifiers
    # runClassifiers(X, Y, k)

    #train and store model
    # cl.createRandomForest(X, Y, '/home/student/PycharmProjects/IsotonicRegression/model/ca_RF.mdl', 50, 1)
    # cl.createRandomForest(X, Y, '/home/student/PycharmProjects/IsotonicRegression/model/as70_RF.mdl', 100, 1)
    cl.createRandomForest(X, Y, '/home/student/PycharmProjects/IsotonicRegression/model/tella09_RF.mdl', 50, 1)

def trainDeepNetwork(url, mdlDumpPath):
    df = pandas.read_csv(url, header=None)
    print df.head(5)
    print df.shape
    print df.describe

    (rows,cols) = df.shape

    X = df.iloc[:, 1:(cols-1)].values
    Y = df.iloc[:, (cols-1)].values #gain
    print "X-head: ", X[:3]
    print "Y-head: ", Y[:3]

    # 1. Visualize data
    # import matplotlib.pyplot as plt
    # df.hist()
    # plt.show()

    #selecting parameters for model
    # rl.tuneRandomForest(X, Y)
    # rl.tuneLassoRegression(X, Y)
    # rl.tuneRidgeRegression(X, Y)
    # rl.tuneLinearSVR(X, Y)
    # rl.tuneSVMRegressor(X, Y)

    if (1):
        mdl = 0;
        # mdl = rl.linearRegression(X, Y)
        # mdl = rl.lassoRegression(X, Y)
        # mdl = rl.ridgeRegression(X, Y)
        mdl = rl.svmRegressor(X, Y)

        print "score: ", mdl.score(X, Y)
        # print "coef: ", mdl.coef_
        # mdl = rl.randomForestRegressor(X, Y)
        # print mdl
        # print "OOB score: ", mdl.oob_score_

        #store model to disk
        from sklearn.externals import joblib
        joblib.dump(mdl, mdlDumpPath)
        print "dumped model to ", mdlDumpPath

def predictGainDeepWalk(mdlDumpPath):
    dir = "/home/student/workspace/GraphSampling/exp/set3/testing/"
    # names = [
    #           "CA-GrQc.txt.dw.r32.50.test",
    #           "CA-GrQc.txt.dw.r32.60.test",
    #           "CA-GrQc.txt.dw.r32.70.test",
    #           "CA-GrQc.txt.dw.r32.80.test",
    #           "CA-GrQc.txt.dw.r32.90.test",
    #           "CA-GrQc.txt.dw.r32.100.test",
    #           "CA-GrQc.txt.dw.r32.110.test",
    #           "CA-GrQc.txt.dw.r32.120.test",
    #           "CA-GrQc.txt.dw.r32.130.test",
    #           "CA-GrQc.txt.dw.r32.140.test",
    #           "CA-GrQc.txt.dw.r32.150.test"
    #         ]
    names = [
        "CA-HepPh.txt.dw.r32.50.test",
        "CA-HepPh.txt.dw.r32.60.test",
        "CA-HepPh.txt.dw.r32.70.test",
        "CA-HepPh.txt.dw.r32.80.test",
        "CA-HepPh.txt.dw.r32.90.test",
        "CA-HepPh.txt.dw.r32.100.test",
        "CA-HepPh.txt.dw.r32.110.test",
        "CA-HepPh.txt.dw.r32.120.test",
        "CA-HepPh.txt.dw.r32.130.test",
        "CA-HepPh.txt.dw.r32.140.test",
        "CA-HepPh.txt.dw.r32.150.test"
    ]
    # names = [
    #     "tella09.txt.dw.r32.50.test",
    #     "tella09.txt.dw.r32.60.test",
    #     "tella09.txt.dw.r32.70.test",
    #     "tella09.txt.dw.r32.80.test",
    #     "tella09.txt.dw.r32.90.test",
    #     "tella09.txt.dw.r32.100.test",
    #     "tella09.txt.dw.r32.110.test",
    #     "tella09.txt.dw.r32.120.test",
    #     "tella09.txt.dw.r32.130.test",
    #     "tella09.txt.dw.r32.140.test",
    #     "tella09.txt.dw.r32.150.test"
    # ]

    for name in names:
        df = pandas.read_csv(dir + name, header=None)
        print df.head(5)
        uid = df.iloc[:,0].values #list of yellow nodes
        print df.shape
        [row, col] = df.shape

        X = df.iloc[:, 1:col].values
        print X

        #load trained model
        from sklearn.externals import joblib
        mdl = joblib.load(mdlDumpPath)
        #predict
        Y_pred = mdl.predict(X)
        print "Y_pred-head:", Y_pred[:5]
        print "Y_pred-tail:", Y_pred[-5:]
        # print "uid:", uid
        # print type(Y_pred)

        # output = np.concatenate((uid.T, Y_pred.T), axis=1)
        c = np.vstack((uid, Y_pred))
        # print c.T
        np.savetxt(dir + name + ".pred" + ".svmE", c.T, fmt='%1.3f')


def main():
    # DISPLAY SETTING
    pandas.set_option('display.max_columns', None)
    np.set_printoptions(precision=3)

    # args = sys.argv

    nodeCentLearning = 1
    if (nodeCentLearning):
        k = 5
        # url = args[1]
        # mdlDumpPath = url + ".dump"
        url = "/home/student/workspace/GraphSampling/exp/set4/nodecen/data/training/brightkite.txt_BFS_fmatrix"
        mdlDumpPath = "/home/student/workspace/GraphSampling/exp/set4/nodecen/data/training/brightkite.txt_BFS_fmatrix.linReg.mdl"
        testNetworkProbing(url, k, mdlDumpPath)

    deepLearning = 0
    if (deepLearning):
        url = "/home/student/workspace/GraphSampling/exp/set3/training/10k/CA-HepPh.txt_dw.r32.label"
        mdlDumpPath = "/home/student/workspace/GraphSampling/exp/set3/training/10k/CA-HepPh.txt_dw.r32.label.svmrbf_E"
        # url = "/home/student/workspace/GraphSampling/exp/data/test_dw/others/probing/CA-HepPh.txt_dw.label"
        # mdlDumpPath = "/home/student/workspace/GraphSampling/exp/data/test_dw/others/probing/CA-HepPh.txt_dw.mdl"
        # url = "/home/student/workspace/GraphSampling/exp/data/test_dw/others/probing/tella09.txt_dw.label"
        # mdlDumpPath = "/home/student/workspace/GraphSampling/exp/data/test_dw/others/probing/tella09.txt_dw.dml"
        # trainDeepNetwork(url, mdlDumpPath)
        predictGainDeepWalk(mdlDumpPath)

    # testDiabeteData()
    # dir = "/home/student/workspace/GraphSampling/exp/data/test/"
    # dir = "/home/student/workspace/GraphSampling/exp/data/as70/"
    # dir = "/home/student/workspace/GraphSampling/exp/data/tella09/"
    #
    # urls = [
    #     # "CA-GrQc.txt_BFS_fmatrix.5k",
    #     # "as70.txt_BFS_fmatrix"
    #     "p2p-Gnutella09.txt_BFS_fmatrix"
    # ]
    #
    # for url in urls:
    #     print "#################################################################"
    #     print url
    #     bFlag = True
    #     bRatio = 0.5
    #     k = 5
    #     testNetworkData(dir + url, k, bFlag, bRatio)
    #     print "#################################################################"

if __name__ == '__main__':
    main()
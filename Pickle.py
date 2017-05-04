from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
import numpy as np
import pandas

def trainModel():
    clf = svm.SVC()
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)
    print type(X)
    print X.shape
    print X[1:10]
    print clf
    joblib.dump(clf, 'svm.pkl')

mdl = 0
def loadModel(mdlPath):

    global mdl
    mdl = joblib.load(mdlPath);
    print mdl
    print "python, model loaded"
    print "score: ", mdl.score
    return 1
    # X = np.array([[7.9, 9.0, 1.4, 0.2]])
    # pred = predict(X)
    # print "python, pred: ", pred
    # return pred

def loadCSV(url, names):
    df = pandas.read_csv(url, names=names)
    return df

def getR2(y_true, y_pred):
    u = ((y_true - y_pred)**2).sum()
    v = ((y_true - y_true.mean())**2).sum()
    R_square = (1 - u/v)
    return R_square

mdl = 0
def checkLoadedModel(mdlPath):

    global mdl
    mdl = joblib.load(mdlPath);
    print mdl
    print "python, model loaded"
    print "score: ", mdl.score

    # load train data
    trainDataURL = "/home/student/workspace/GraphSampling/exp/set3/training/10k/CA-GrQc.txt_dw.r32.label"
    # trainDataURL = "/home/student/workspace/GraphSampling/exp/set3/test"
    df = pandas.read_csv(trainDataURL, header=None)
    print df.head(5)
    print df.shape
    print df.describe

    (rows, cols) = df.shape

    x_test = df.iloc[:, 1:(cols-1)].values
    y_true = df.iloc[:, (cols-1)].values

    print "X_test: ", x_test[:3]
    print "Y_test: ", y_true[:3]

    y_pred = mdl.predict(x_test)
    print "Y_pred: ", y_pred

    R2 = getR2(y_true, y_pred)
    print "R2: ", R2

    # print "Describe: \n ", dfSel.describe()


    # call predict

    # get R^2 score

def predict(X):
    # print "python, predicting..."
    X = np.reshape(X, (1, -1))
    print "X: ", X
    pred = mdl.predict(X);
    return pred

if __name__ == "__main__":
    # trainModel()
    # mdl = "/home/student/PycharmProjects/IsotonicRegression/model/ca_RF.mdl"
    # X = np.array([[1, 0.000307503, 0.0304251, 0.0192794, 0, 2.26, 25.9898, 172.304]])
    # mdl = "/home/student/workspace/GraphSampling/exp/data/as70/as70.txt.mdl"
    # mdl = "/home/student/workspace/GraphSampling/exp/nodecen/data/CA-GrQc.txt.11k.mdl"
    # mdl = "/home/student/workspace/GraphSampling/exp/set3/nodecen/data/10k/CA-GrQc.txt_BFS_fmatrix.10k.SVM.mdl"
    # mdl = "/home/student/workspace/GraphSampling/exp/set2/training/"
    # loadModel(mdl)

    #validate trained model:
    mdl = "/home/student/workspace/GraphSampling/exp/set3/training/10k/CA-GrQc.txt_dw.r32.label.svmrbf_C"
    # checkLoadedModel(mdl)

    loadModel(mdl)
    X_test = np.array([-0.007451,0.396457,-0.417461,0.043394,0.788562,-0.340413,-0.665572,-0.117094,-0.174358,-0.720543,0.076428,0.018788,0.067205,0.192695,0.298946,0.353858,0.016921,-0.258605,-0.004329,0.644218,0.207449,0.559737,0.522373,0.166972,0.356881,-0.043898,0.338675,0.434227,0.005970,0.127622,0.460111,-0.171047])
    X_train = np.array([0.343292,-0.318907,0.207914,0.909210,0.028058,-0.294983,0.137540,-0.336732,0.260838,0.054268,-0.329810,1.012996,0.435192,0.123494,-0.168681,-0.473601,0.076832,-0.037266,0.170798,-0.244204,0.269922,0.633861,-0.339028,0.493054,-0.168364,-0.134028,-0.254174,0.229308,-0.029540,0.878671,-0.743707,0.938495])
    # print "X: ", X
    print "X_pred: ", predict(X_train)

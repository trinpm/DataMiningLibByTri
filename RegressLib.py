import numpy as np

def linearRegression(train, label):
    print "Linear Regression Model"
    from sklearn import linear_model
    linearReg = linear_model.LinearRegression(normalize=False)
    linearReg.fit(train, label)
    return linearReg

def logisticRegression(train, label):
    """Logistic Regression Model"""
    from sklearn import linear_model
    logisticReg = linear_model.LogisticRegression(C=1e5)
    logisticReg.fit(train, label)
    return logisticReg

def ridgeRegression(train, label):
    print "Ridge Regression Model"
    from sklearn import linear_model
    ridgeReg = linear_model.Ridge(alpha=1)
    ridgeReg.fit(train, label)
    return ridgeReg

def tuneLinearSVR(train, label):
    print "tunning SVM-linear..."
    from sklearn.svm import SVR
    alpha = [50, 100, 1000, 10000]
    for a in alpha:
        print "tuneLinearSVR with alpha = ", a
        mdl = SVR(kernel="linear", C=a)
        mdl.fit(train, label)
        print "score = {}, alpha = {} ".format(mdl.score(train, label), a)
        print "coef: ", mdl.coef_

def linearSVR(train, label):
    print "tuneLinearSVR"
    from sklearn.svm import SVR
    mdl = SVR(kernel="linear")
    # print len(train)
    # print train[:3]
    mdl.fit(train, label)
    # print "score = {}, alpha = {} ".format(mdl.score(train, label))
    # print "coef: ", mdl.coef_
    return mdl

def tuneRidgeRegression(train, label):
    from sklearn import linear_model as lm
    alpha = [0, 0.1, 0.2, 0.5, 0.8, 1, 10]
    for a in alpha:
        mdl = lm.Ridge(alpha=a)
        mdl.fit(train, label)
        print "score: ", mdl.score(train, label)

def lassoRegression(train, label):
    print "Lasso Regression Model"
    from sklearn import linear_model
    lassoReg = linear_model.Lasso(alpha=1.0)
    lassoReg.fit(train, label)
    return lassoReg

def tuneLassoRegression(train, label):
    print "tunning LassoRegression..."
    from sklearn import linear_model as lm
    alpha = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    for a in alpha:
        mdl = lm.Lasso(alpha=a)
        mdl.fit(train, label)
        print "score = {}, alpha = {}".format(mdl.score(train, label), a)
        print "coef = ", mdl.coef_

def randomForestRegressor(train, label):
    """Randomforest Regressor"""
    from sklearn.ensemble import RandomForestRegressor
    mdl = RandomForestRegressor(n_estimators=30, oob_score=True)
    mdl.fit(train, label)
    return mdl

def svmRegressor(train, label):
    from sklearn.svm import SVR
    mdl = SVR(kernel='rbf', C=100, gamma=10)
    mdl.fit(train, label)
    return mdl

def tuneSVMRegressor(train, label):
    print "tunning SVM-RBF..."
    from sklearn.svm import SVR
    C = [1e-4, 1e-3, 1e-2,  1e-1, 1e0, 1e1, 1e2, 1e3, 1e4 ]
    gamma = [ 0.1, 1, 10, 100 ]
    for c in C:
        for g in gamma:
            mdl = SVR(kernel='rbf', C=c, gamma=g)
            mdl.fit(train, label)
            print "score = {}, c = {}, gamma = {}".format(mdl.score(train,label), c, g)

def tuneRandomForest(train, label):
    from sklearn.ensemble import RandomForestRegressor

    n_estimators = [1, 10, 20, 30, 40, 50]
    leaf_sizes = [1, 50, 100, 200, 500, 600, 700]

    for n in n_estimators:
        for l in leaf_sizes:
            mdl = RandomForestRegressor(n_estimators=n, min_samples_leaf=l, oob_score=True)
            mdl.fit(train, label)
            print "oob score = {}, n_estimators = {}, min_samples_leaf = {}".format(mdl.oob_score_, n ,l)

def main():
    print "Regression Library"
    from sklearn import datasets

    diabetes = datasets.load_diabetes()

    print "dimension = ", diabetes.data.ndim
    print "shape = ", diabetes.data.shape

    # get training data
    diabetes_X_train = diabetes.data[:-20]  #select the first 20 rows
    diabetes_X_test = diabetes.data[-20:]   #select the last 20 rows
    # get test data
    diabetes_Y_train = diabetes.target[:-20]
    diabetes_Y_test = diabetes.target[-20:]

    lin = linearRegression(diabetes_X_train, diabetes_Y_train);
    log = logisticRegression(diabetes_X_train, diabetes_Y_train);
    rid = ridgeRegression(diabetes_X_train, diabetes_Y_train);
    lasso = lassoRegression(diabetes_X_train, diabetes_Y_train);

    MSE = np.mean((lin.predict(diabetes_X_test) - diabetes_Y_test) ** 2)
    print 'MSE by linear regression model: ', MSE

    MSE = np.mean((log.predict(diabetes_X_test) - diabetes_Y_test) ** 2)
    print 'MSE by logistic regression model: ', MSE

    MSE = np.mean((rid.predict(diabetes_X_test) - diabetes_Y_test) ** 2)
    print 'MSE by ridge regression model: ', MSE

    MSE = np.mean((lasso.predict(diabetes_X_test) - diabetes_Y_test) ** 2)
    print 'MSE by lasso regression model: ', MSE


if __name__ == '__main__':
    main()
import pandas
import ClassifyLib as cl
import ParameterTuning as pt
import Utils as ut
import sys
import time
import numpy as np
import Utils as ut

def loadData(url, d):
    df = pandas.read_csv(url, delimiter=d)
    return df


def getDict(keys, values):
    dict = {}
    for i in range(0, len(keys)):
        dict[keys[i]] = values[i]
    return dict


def getCorrelation(X, Y, name):
    corr=[]
    [row, col] = X.shape
    print "row={}, col={}".format(row,col)
    for i in range(0, col):
        x = X[:,i]
        # print "x: ", x
        # print "Y: ", Y
        if name == "pearson":
            c = ut.getPearsonCorrelation(x, Y)
        elif name == "spearman":
            c= ut.getSpearmanCorrelation(x, Y)

        # print "c: ", c
        # print "type of c: ", type(c)
        corr.append(c)

    r_value = []
    p_value = []
    for i in range(0, len(corr)):
        tuple = corr[i]
        r_value.append(tuple[0])
        p_value.append(tuple[1])
        print str(r_value[i]) + ", " + str(p_value[i])

    return [r_value, p_value]


def selectData(df, label, begin, end):
    print df.head(5)

    headers = df.columns.values
    headers_train = headers[begin - 1:end]
    print "header: ", headers_train

    # select columns to be trained:
    data = df.values
    [row, col] = data.shape
    print "Dimension: ", data.shape

    X = data[:, (begin - 1):(end)]  # [a,b)
    Y = data[:, (label - 1)]

    print "X (first few): ", X[:5]
    print "Y (first few): ", Y[:5]

    return [X, Y, headers_train]


def runAnalysis(X, Y, C_logistic, C_linear_svm):

    kfold = 5

    # parameter tuning:
    # pt.tuneParametersForSVM(X, Y, kfold)

    # run classification with logistic regression
'''
    print "logistic regression: "
    coefs = cl.logisticRegression(X, Y, kfold, C_logistic)
    print coefs[0:10]
    coefs = np.absolute(coefs)
    print coefs[0:10]
    print headers_train[0:10]
    print "coefs: ", coefs
    write2cols2file(url + ".logisticReg", headers_train, coefs)



    print "linear SVM: "
    coefs = cl.SVM_lin_regu(X, Y, kfold, C_linear_svm)
    coefs = np.absolute(coefs)
    print "coefs: ", coefs
    write2cols2file(url + ".linearSVM", headers_train, coefs)
'''


def main():
    print "alcoholic"

    # load alcoholic data
    # url = "/home/student/trinpm/SPECS/bio/alcohol/data/PFC_NAC_Vla_Spearman_cons.csv"
    url = "/home/student/trinpm/SPECS/bio/alcohol/data/PFC_gene.txt"
    # url = "/home/student/trinpm/SPECS/bio/alcohol/data/PFC_miRNA.txt"
    # url = "/home/student/trinpm/SPECS/bio/alcohol/data/NAC_gene.txt"
    # url = "/home/student/trinpm/SPECS/bio/alcohol/data/NAC_miRNA.txt"
    # url = "/home/student/trinpm/SPECS/bio/alcohol/data/NAC_lncRNA.txt"

    # args = sys.argv
    # url = args[1]

    if "PFC_gene" in url:
        label = 6
        a = 24
        b = 22301
        C_logistic = 0.001
        C_linear_svm = 0.001
        label_drink_cons = 8

    elif "PFC_miRNA" in url:
        label = 8
        a = 19
        b = 1751
        C_logistic = 0.1
        C_linear_svm = 0.1
        label_drink_cons = 6

    elif "NAC_gene" in url:
        label = 5
        a = 9
        b = 22305
        C_logistic = 0.0001
        C_linear_svm = 0.001
        label_drink_cons = 3

    elif "NAC_miRNA" in url:
        label = 11
        a = 23
        b = 1755
        C_logistic = 0.1
        C_linear_svm = 0.1
        label_drink_cons = 3

    elif "NAC_lncRNA" in url:
        label = 9
        a = 18
        b = 43701
        C_logistic = 10
        C_linear_svm = 0.001
        label_drink_cons = 4

    elif "test" in url:
        label = 5
        a = 1
        b = 4
        C_logistic = 1
        C_linear_svm = 1

    print "url: ", url

    start_time = time.time()
    # df = loadData(url, ",") # test data
    df = loadData(url, "\t")  # bio data
    [X, Y, headers] = selectData(df, label, a, b) #using Diagnosis as label
    # [X, Y, headers] = selectData(df, label_drink_cons, a, b) #using Alcohol Consumption as label

    # get Correlations
    # corr_name = "spearman"
    # [r_value, p_value] = getCorrelation(X, Y, corr_name)
    # ut.write3cols2file(url + ".alc_consump." + corr_name, headers, r_value, p_value)

    # df = loadData(url, "\t")  # alcohol data
    runAnalysis(X, Y, C_logistic, C_linear_svm)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
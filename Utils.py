

from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import pandas
import scipy.stats as st

def balanceData(X, Y, r):
    from imblearn.over_sampling import SMOTE
    print('Original dataset shape {}'.format(Counter(Y)))
    sm = SMOTE(ratio=r) # (#minority data)/(#majority data)
    X_new, Y_new = sm.fit_sample(X, Y)
    print('Resampled dataset shape {}'.format(Counter(Y_new)))
    return X_new, Y_new

def balanceDataRandom(X, Y):
    print "size of X={}, size of Y={}".format(X.shape, Y.shape)
    from collections import Counter
    print('Original dataset shape {}'.format(Counter(Y)))
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_new, Y_new = ros.fit_sample(X, Y)
    print('Resampled dataset shape {}'.format(Counter(Y_new)))
    print "size of X_res={}, size of Y_res={}".format(X_new.shape, Y_new.shape)
    return X_new, Y_new

def minMaxScale(df):
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    print type(df)
    return df

def loadCSV(url, d):
    df = pandas.read_csv(url, delimiter=d)
    return df

def getPearsonCorrelation(x, y):
    return st.pearsonr(x, y)

def getSpearmanCorrelation(x, y):
    return st.spearmanr(x, y)

def write2cols2file(fname, vars, coefs):
    out = open(fname, 'w')
    if len(vars) == len(coefs):
        for i in range(0, len(vars)):
            # line = vars[i] + "\t" + str(coefs[i]) + "\n"
            out.write("%s\t%.8f\n" % (vars[i], coefs[i]))
    else:
        print "error!"
    out.close()

def write3cols2file(fname, col1, col2, col3):
    out = open(fname, 'w')
    if len(col1) == len(col2) == len(col3):
        for i in range(0, len(col1)):
            # line = vars[i] + "\t" + str(coefs[i]) + "\n"
            out.write("%s\t%.2f\t%.2f\n" % (col1[i], col2[i], col3[i]))
    else:
        print "error!"
    out.close()

def writeList2file(fname, data):
    out = open(fname, 'w')
    for i in range(0, len(data)):
        out.write("%s\n" % (data[i]))
    out.close()
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
import numpy as np

def recursiveFeatureElimination(X, Y, _k):
    model = LogisticRegression()
    rfe = RFE(model, _k/2)
    rfe = rfe.fit(X, Y)
    # summarize the selection of the attributes
    # print "support:\n", rfe.support_
    print "ranking:\n", rfe.ranking_

def selectKBest(X, Y, _k):
    kbest = SelectKBest(score_func=chi2, k = _k)
    fit = kbest.fit(X, Y)
    print "ranking by chi2:\n", fit.scores_

    # kbest = SelectKBest(score_func=f_classif, k=5)
    # fit = kbest.fit(X, Y)
    # print "ranking by f_classif:\n", fit.scores_

    kbest = SelectKBest(score_func=mutual_info_classif, k= _k)
    fit = kbest.fit(X, Y)
    print "ranking by mutual_info_classif:\n", fit.scores_

def extraTrees(X, Y):
    model = ExtraTreesClassifier(n_estimators=100)
    model.fit(X, Y)
    #display the relative importance of each attribute
    print "ranking:\n", model.feature_importances_


import numpy as np
import pandas as pd
import tensorflow as tf
import alignment
import alignment_cwe
import crf
import bilstm_cbow
import bilstm_cwe
import util
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class GBDT1():
    def __init__(self):
        self.n_estimators=30
        self.learning_rate=0.1
        self.sub_sample=0.8
        self.loss_type="deviance"

        self.gbdt=GradientBoostingClassifier(
            loss=self.loss_type,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.sub_sample
        )

    def fit(self,X_train,y_train,X_test,y_test):
        self.gbdt.fit(X=X_train,y=y_train)
        pred=self.gbdt.predict(X=X_test)
        print(pred.shape)
        print("accracy:",accuracy_score(y_true=y_test,y_pred=pred))
        print("f1-score:",f1_score(y_true=y_test,y_pred=pred,average=None))


    def pred(self,X):
        pass


if __name__=="__main__":
    print("loading data....")
    #training data
    X_train_crf,labels_train,preds_train_crf=util.extractProb(file="../result/crf/crf_prob_train.txt")
    X_train_alignment=util.extractProb2(file="../result/alignment/alignment_prob_train.txt")
    X_train=np.concatenate((X_train_crf,X_train_alignment),axis=1)

    # test data
    X_test_crf, labels_test, preds_test_crf = util.extractProb(file="../result/crf/crf_prob_test.txt")
    X_test_alignment = util.extractProb2(file="../result/alignment/alignment_prob_test.txt")
    X_test = np.concatenate((X_test_crf, X_test_alignment), axis=1)

    print("run model....")
    model=GBDT1()
    model.fit(X_train=X_train,y_train=labels_train,X_test=X_test,y_test=labels_test)



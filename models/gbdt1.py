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


    def fit(self,X_train,y_train,X_valid,y_valid):
        self.gbdt.fit(X=X_train,y=y_train)
        pred=self.gbdt.predict(X=X_valid)
        print(pred.shape)
        print("accracy:",accuracy_score(y_true=y_valid,y_pred=pred))
        print("f1-score:",f1_score(y_true=y_valid,y_pred=pred,average=None))


    def pred(self,X):
        pass


if __name__=="__main__":
    print("loading data....")
    X_crf,labels=util.extractProb(file="../result/crf/crf_prob_train.txt")
    X_alignment=util.extractProb2(file="../result/alignment/alignment_prob_train.txt")
    #print(X_crf.shape)
    #print(labels.shape)
    #print("labels",labels)
    #print(X_alignment.shape)
    X=np.concatenate((X_crf,X_alignment),axis=1)
    #print(X.shape)
    X_train=X[:700000]
    X_valid=X[700000:]
    #print(X_train.shape)
    #print(X_valid.shape)
    labels_train=labels[:700000]
    labels_valid=labels[700000:]

    print("run model....")
    model=GBDT1()
    model.fit(X_train=X_train,y_train=labels_train,X_valid=X_valid,y_valid=labels_valid)



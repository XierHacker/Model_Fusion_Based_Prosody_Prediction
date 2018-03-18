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
        self.learning_rate=0.08
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

    def deMask(self):
        pass

    def pred(self,X):
        pass

#depadding and will reduce dimension
def mask(length,X):
    list = []
    for i in range(length.shape[0]):
        sentenece_len = length[i]
        for j in range(sentenece_len):
            list.append(X[i, j])
    return np.array(list,dtype=np.int32)

def onehot(array):
    a=np.zeros(shape=(array.shape[0],37),dtype=np.int32)
    for i in range(array.shape[0]):
        a[i,array[i]-1]=1
    return a

if __name__=="__main__":

    print("loading data....")
    #training data
    # pw 为了获取长度信息
    #df_train_pw = pd.read_pickle(path="../data/dataset/pw_summary_train.pkl")
    #len_train = np.asarray(list(df_train_pw['sentence_len'].values))

    #X_train_crf,labels_train,preds_train_crf=util.extractProb(file="../result/crf/crf_prob_train.txt")
    #X_train_alignment=util.extractProb2(file="../result/alignment/alignment_prob_train.txt")
    #X_train_cnn = util.extractProb2(file="../result/cnn/cnn_prob_train.txt")
    #print("X_train_cnn.shape",X_train_cnn.shape)

    #pos_train=util.readExtraInfo(file="../data/dataset/pos_train_tag.txt")
    #pos_train_masked=mask(length=len_train,X=pos_train)
    #print(pos_train_masked.shape)
    #pos_train_onehot=onehot(pos_train_masked)
    #print(pos_train_onehot.shape)
    #X_train=np.concatenate((X_train_cnn,X_train_alignment,pos_train_onehot),axis=1)

    #valid data
    df_valid_pw = pd.read_pickle(path="../data/dataset/pw_summary_valid.pkl")
    len_valid = np.asarray(list(df_valid_pw['sentence_len'].values))
    X_valid_crf, labels_valid, preds_valid_crf = util.extractProb(file="../result/crf/crf_prob_valid.txt")
    X_valid_alignment = util.extractProb2(file="../result/alignment/alignment_prob_valid_epoch5.txt")
    X_valid_cnn = util.extractProb2(file="../result/cnn/cnn_prob_valid_epoch5.txt")
    X_valid_attention=util.extractProb2(file="../result/attention/attention_prob_valid_epoch4.txt")
    X_valid_bilstm=util.extractProb2(file="../result/bilstm/bilstm_prob_valid_epoch3.txt")

    pos_valid = util.readExtraInfo(file="../data/dataset/pos_valid_tag.txt")
    pos_valid_masked = mask(length=len_valid, X=pos_valid)
    print(pos_valid_masked.shape)
    pos_valid_onehot = onehot(pos_valid_masked)
    print(pos_valid_onehot.shape)
    X_valid = np.concatenate(
        (X_valid_crf,X_valid_cnn, X_valid_alignment,X_valid_attention, X_valid_bilstm,pos_valid_onehot),
        axis=1
    )

    # test data
    df_test_pw = pd.read_pickle(path="../data/dataset/pw_summary_test.pkl")
    len_test = np.asarray(list(df_test_pw['sentence_len'].values))
    X_test_crf, labels_test, preds_test_crf = util.extractProb(file="../result/crf/crf_prob_test.txt")
    X_test_alignment = util.extractProb2(file="../result/alignment/alignment_prob_test_epoch5.txt")
    X_test_cnn = util.extractProb2(file="../result/cnn/cnn_prob_test_epoch5.txt")
    X_test_attention = util.extractProb2(file="../result/attention/attention_prob_test_epoch4.txt")
    X_test_bilstm=util.extractProb2(file="../result/bilstm/bilstm_prob_test_epoch3.txt")

    pos_test = util.readExtraInfo(file="../data/dataset/pos_test_tag.txt")
    pos_test_masked = mask(length=len_test, X=pos_test)
    print(pos_test_masked.shape)
    pos_test_onehot = onehot(pos_test_masked)
    print(pos_test_onehot.shape)
    X_test = np.concatenate(
        (X_test_crf,X_test_cnn, X_test_alignment, X_test_attention,X_test_bilstm,pos_test_onehot),
        axis=1
    )

    print("run model....")
    model=GBDT1()
    #model.fit(X_train=X_train, y_train=labels_train, X_test=X_valid, y_test=labels_valid)
    model.fit(X_train=X_valid,y_train=labels_valid,X_test=X_test,y_test=labels_test)



import numpy as np
import pandas as pd
import tensorflow as tf
import parameter
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

#compute accuracy,precison,recall and f1
def eval(y_true,y_pred):
    #accuracy
    accuracy=accuracy_score(y_true=y_true,y_pred=y_pred)
    #f1-score
    f_1=f1_score(y_true=y_true,y_pred=y_pred,average=None)
    return accuracy,f_1

def getTag2(preds_pw,preds_pph):
    # get complex "#" index
    length = preds_pw.shape[0]
    complex = np.array([preds_pph, preds_pw])
    arg = np.argmax(complex, axis=0)
    # print("arg:\n", arg)
    for i in range(length):
        if arg[i] == 0:
            if complex[0, i] == 1:
                arg[i] = 4
            else:
                arg[i] = 0
        if arg[i] == 1:
            if complex[1, i] == 1:
                arg[i] = 2
            else:
                arg[i] = 0
    arg = (arg / 2).astype(dtype=np.int32)
    return arg


#recover to .txt format
def recover2(X,preds_pw,preds_pph,filename):
    arg=getTag2(preds_pw,preds_pph)
    arg=np.reshape(arg,newshape=(-1,parameter.MAX_SENTENCE_SIZE))   #[test_size,max_sentence_size]
    print("arg.shape",arg.shape)
    print("arg:\n", arg)
    #get id2words
    df_words_ids = pd.read_csv(filepath_or_buffer="../data/dataset/temptest/words_ids.csv", encoding="utf-8")
    #print(df_words_ids.head(5))
    id2words = pd.Series(data=df_words_ids["words"].values, index=df_words_ids["id"].values)
    #print(id2words[2])
    doc=""
    for i in range(X.shape[0]):
        sentence=""
        for j in range(X.shape[1]):
            if(X[i][j])==0:
                break;
            else:
                sentence+=id2words[X[i][j]]
                if(arg[i][j]!=0):
                    sentence+=("#"+str(arg[i][j]))
        sentence+="\n"
        doc+=sentence
    f=open(filename,mode="w",encoding="utf-8")
    f.write(doc)
    f.close()

#read extra info form file,like pos info of word,or position info etc...
#trans to one-hot format
def readExtraInfo(file):
    f = open(file=file, encoding="utf-8")
    lines = f.readlines()
    #print("lines numbers:",len(lines))
    X=np.zeros(shape=(len(lines),parameter.MAX_SENTENCE_SIZE),dtype=np.int32)
    i = 0
    for line in lines:
        # print(line)
        line = line.strip()
        line_list = line.split(sep=" ")
        # print(line_list)
        j = 0
        for id in line_list:
            X[i, j] = id
            j += 1
        i += 1
    #print(X)
    #print("word_id_train.shape", X.shape)
    return X

    #one-hot encoder


if __name__ =="__main__":
    print("read extra info test:")
    #readExtraInfo(file="./data/dataset/pos_train_tag.txt")
    #readExtraInfo(file="./data/dataset/pos_test_tag.txt")

    #readExtraInfo(file="./data/dataset/length_train_tag.txt")
    #readExtraInfo(file="./data/dataset/length_test_tag.txt")



'''
def getTag(preds_pw,preds_pph,preds_iph):
    # get complex "#" index
    length = preds_pw.shape[0]
    complex = np.array([preds_iph, preds_pph, preds_pw])
    arg = np.argmax(complex, axis=0)
    # print("arg:\n", arg)
    for i in range(length):
        if arg[i] == 0:
            if complex[0, i] == 2:
                arg[i] = 6
            else:
                arg[i] = 0
        if arg[i] == 1:
            if complex[1, i] == 2:
                arg[i] = 4
            else:
                arg[i] = 0
        if arg[i] == 2:
            if complex[2, i] == 2:
                arg[i] = 2
            else:
                arg[i] = 0
    arg = (arg / 2).astype(dtype=np.int32)
    return arg

#recover to original result
def recover(X,preds_pw,preds_pph,preds_iph,filename):
    #shape of arg:[test_size,max_sentence_size]
    arg=np.reshape(arg,newshape=(-1,parameter.MAX_SENTENCE_SIZE))
    #print("arg.shape",arg.shape)
    #print("arg:\n", arg)
    #get id2words
    df_words_ids = pd.read_csv(filepath_or_buffer="./dataset/temptest/words_ids.csv", encoding="utf-8")
    #print(df_words_ids.head(5))
    id2words = pd.Series(data=df_words_ids["words"].values, index=df_words_ids["id"].values)
    #print(id2words[2])
    doc=""
    for i in range(X.shape[0]):
        sentence=""
        for j in range(X.shape[1]):
            if(X[i][j])==0:
                break;
            else:
                sentence+=id2words[X[i][j]]
                if(arg[i][j]!=0):
                    sentence+=("#"+str(arg[i][j]))
        sentence+="\n"
        doc+=sentence
    f=open(filename,mode="w",encoding="utf-8")
    f.write(doc)
    f.close()
'''
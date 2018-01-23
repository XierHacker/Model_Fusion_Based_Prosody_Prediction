import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import parameter

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

#def charBasedTag2(X,preds_pw,preds_pph):
    # get id2words
#    df_words_ids = pd.read_csv(filepath_or_buffer="../data/dataset/temptest/words_ids.csv", encoding="utf-8")
    # print(df_words_ids.head(5))
#    id2words = pd.Series(data=df_words_ids["words"].values, index=df_words_ids["id"].values)
    # print(id2words[2])

    #


#recover to original result
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


if __name__ =="__main__":
    #测试
    a=np.array([1,2,3,4,0,5,6,7,1,1,2,1,0])
    print(a)
    result=binarize(sequence=a,positive_value=1)
    print(result)
    print(a)



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
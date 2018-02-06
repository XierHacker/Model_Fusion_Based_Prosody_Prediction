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

#从得到的prob_pw和prob_pph得到总的prob,并保存
def writeProb(prob_pw,prob_pph,outFile):
    f=open(file=outFile,mode="a+",encoding="utf-8")
    for i in range(prob_pw.shape[0]):
        prob_0=prob_pw[i,0]*prob_pph[i,0]
        prob_1=prob_pw[i,1]*prob_pph[i,0]
        prob_2=prob_pw[i,0]*prob_pph[i,1]+prob_pw[i,1]*prob_pph[i,1]
        s=str(prob_0)+" "+str(prob_1)+" "+str(prob_2)+"\n"
        f.write(s)
    f.close()

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
    #print("arg.shape",arg.shape)
    #print("arg:\n", arg)
    #get id2words
    df_words_ids = pd.read_csv(filepath_or_buffer="../data/dataset/words_ids.csv", encoding="utf-8")
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

#read extra information from file,like pos info of word,or position info etc...
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
    return X


#读取预训练的embeddings
def readEmbeddings(file):
    f=open(file=file,encoding="utf-8")
    lines=f.readlines()
    #first row is info
    info=lines[0].strip()
    info_list=info.split(sep=" ")
    vocab_size=int(info_list[0])
    embedding_dims=int(info_list[1])
    embeddings=np.zeros(shape=(vocab_size+1,embedding_dims),dtype=np.float32)
    for i in range(1,vocab_size+1):
        embed=lines[i].strip()
        embed_list=embed.split(sep=" ")
        for j in range(1,embedding_dims+1):
            embeddings[i][j-1]=embed_list[j]
    #print(embeddings.shape)
    return embeddings

#返回字增强之后的word-embeddings
def getCWE(word_embed_file,char_embed_file):
    word_embeddings=readEmbeddings(file=word_embed_file)
    print("shape of word_embeddings:",word_embeddings.shape)
    char_embeddings=readEmbeddings(file=char_embed_file)
    print("shape of char_embeddings:",char_embeddings.shape)

    #load id-word df
    df_words_ids = pd.read_csv(filepath_or_buffer="../data/dataset/words_ids.csv", encoding="utf-8")
    id2words = pd.Series(data=df_words_ids["words"].values, index=df_words_ids["id"].values)

    #load id-char df
    df_chars_ids = pd.read_csv(filepath_or_buffer="../data/dataset/chars_ids.csv", encoding="utf-8")
    chars2id = pd.Series(data=df_chars_ids["id"].values, index=df_chars_ids["chars"].values)

    for i in range(1,word_embeddings.shape[0]):
        #print(id2words[i])
        word=id2words[i]
        sum_char_embeddings=np.zeros(shape=(128,),dtype=np.float32)
        for char in word:
            char_id=chars2id[char]
            sum_char_embeddings+=char_embeddings[char_id]
        sum_char_embeddings/=len(word)
        word_embeddings[i]+=sum_char_embeddings
    cwe=word_embeddings/2
    return cwe


#从crf结果文件中抽取概率,并且返回ndarray类型
def extractProb(file):
    probs=[]
    labels=[]
    preds=[]
    f=open(file=file,encoding="utf-8")
    lines=f.readlines()
    for line in lines:
        line=line.strip()
        if line!="":
            if line[0]!="#":
                prob = []
                #print(line)
                #print(line[0])
                line_list = line.split(sep="\t")
                l_0 = line_list[9].split(sep="/")
                prob.append(float(l_0[1]))
                l_1 = line_list[10].split(sep="/")
                prob.append(float(l_1[1]))
                l_2 = line_list[11].split(sep="/")
                prob.append(float(l_2[1]))
                #print(prob)
                probs.append(prob)
                labels.append(float(line_list[7]))
                preds.append(float(line_list[8].split(sep="/")[0]))
    #print("len of probs:",probs[0])
    probs_nd=np.array(probs,dtype=np.float32)
    labels_nd=np.array(labels,dtype=np.int32)
    preds_nd=np.array(preds,dtype=np.int32)
    #print("shape of prob_nd",probs_nd.shape)
    return probs_nd,labels_nd,preds_nd

def extractProb2(file):
    probs=[]
    result=[]
    f=open(file=file,encoding="utf-8")
    lines=f.readlines()
    for line in lines:
        line=line.strip()
        prob = []
        #print(line)
        #print(line[0])
        line_list = line.split(sep=" ")
        #print(line_list)
        l_0 = line_list[0]
        prob.append(float(l_0))
        l_1 = line_list[1]
        prob.append(float(l_1))
        l_2 = line_list[2]
        prob.append(float(l_2))
        #print(prob)
        probs.append(prob)
    #print("len of probs:",probs[0])
    probs_nd=np.array(probs,dtype=np.float32)
    #print("shape of prob_nd",probs_nd.shape)
    #print(probs_nd.dtype)
    return probs_nd


if __name__ =="__main__":
    #print("read extra info test:")
    #readExtraInfo(file="./data/dataset/pos_train_tag.txt")
    #readExtraInfo(file="./data/dataset/pos_test_tag.txt")
    #readExtraInfo(file="./data/dataset/length_train_tag.txt")
    #readExtraInfo(file="./data/dataset/length_test_tag.txt")
    #readEmbeddings(file="./data/embeddings/word_vec.txt")
    #readEmbeddings(file="./data/embeddings/char_vec.txt")
    #getCWE(word_embed_file="./data/embeddings/word_vec.txt",char_embed_file="./data/embeddings/char_vec.txt")
    print("CRF")
    prob,labels,preds=extractProb(file="./result/crf/crf_prob_test.txt")
    #print("prob.shape",prob.shape)
    #print("labels.shape", labels.shape)
    #print("preds.shape",preds.shape)
    p1, f1 = eval(y_true=labels, y_pred=preds)
    print("accuracy:", p1)
    print("f1-score:", f1)

    print("Alignment")
    prob_align=extractProb2(file="./result/alignment/alignment_prob_test.txt")
    #print("prob_align.shape",prob_align.shape)
    #print("prob_align:",prob_align)
    preds_align=np.argmax(prob_align,axis=-1,)
    #print(preds_align.shape)
    #print(preds_align)
    p2, f2 = eval(y_true=labels, y_pred=preds_align)
    print("accuracy:",p2)
    print("f1-score:",f2)





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
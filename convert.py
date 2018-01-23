'''
    trans .utf8 files to normal tagged files
'''

import os
import pandas as pd
import numpy as np
import re

#trans to prosody tagged format
def toProsody(inFile,outFile):
    f_in=open(file=inFile,encoding="utf-8")
    doc=""
    lines=f_in.readlines()
    for line in lines:
        #print("line:",line)
        line=line.strip()
        line_list=line.split(sep="\t")
        #print(line_list)
        if(line_list[0]!=""):
            doc+=(line_list[0]+"#"+line_list[-1])
        else:
            doc+="\n"
    #print(doc)
    f_out=open(file=outFile,mode="w",encoding="utf-8")
    f_out.write(doc)
    f_out.close()

def toProsody2(inFile,outFile):
    f_in=open(file=inFile,encoding="utf-8")
    doc=""
    lines=f_in.readlines()
    for line in lines:
        #print("line:",line)
        line=line.strip()
        line_list=line.split(sep="\t")
        #print(line_list)
        if(line_list[0]!=""):
            doc+=(line_list[0]+"#"+line_list[-2])
        else:
            doc+="\n"
    #print(doc)
    f_out=open(file=outFile,mode="w",encoding="utf-8")
    f_out.write(doc)
    f_out.close()


'''
#生成分词格式的文本,
def toWords(inFile_train,inFile_valid,inFile_test):
    #---------------------------------------生成单词列表--------------------------------------------#
    f_train_in = open(file=inFile_train, encoding="utf-8")
    f_valid_in = open(file=inFile_valid, encoding="utf-8")
    f_test_in = open(file=inFile_test, encoding="utf-8")

    words=[]
    #收集所有单词
    lines_train = f_train_in.readlines()
    for line_train in lines_train:
        line_train = line_train.strip()
        line_train_list = line_train.split(sep="\t")
        if (line_train_list[0] != ""):
            words.append(line_train_list[0])
    f_train_in.close()

    lines_valid = f_valid_in.readlines()
    for line_valid in lines_valid:
        line_valid = line_valid.strip()
        line_valid_list = line_valid.split(sep="\t")
        if (line_valid_list[0] != ""):
            words.append(line_valid_list[0])
    f_valid_in.close()

    lines_test = f_test_in.readlines()
    for line_test in lines_test:
        line_test = line_test.strip()
        line_test_list = line_test.split(sep="\t")
        if (line_test_list[0] != ""):
            words.append(line_test_list[0])
    f_test_in.close()


    #print(words)
    print("origin len of words:",len(words))
    sr_allwords = pd.Series(data=words)             # 2.列表做成pandas的Series
    words = (sr_allwords.value_counts()).index      # 词列表.统计每个字出现的频率,同时相当于去重复,得到字的集合(这里还是Serieas的index对象)
    print(words)
    print("len of cleaned:",words.shape)
    words_id = range(1, len(words) + 1)             # 字的id列表,从1开始，因为准备把0作为填充值

    # words以及对应的id组件
    df_words_ids=pd.DataFrame(data={"words": words, "id": words_id})
    df_words_ids. to_csv(path_or_buf="words_ids.csv", index=False, encoding="utf_8")

    words2id = pd.Series(data=df_words_ids["id"].values, index=df_words_ids["words"].values)
    id2words = pd.Series(data=df_words_ids["words"].values, index=df_words_ids["id"].values)

    print("words2id:\n",words2id.head(10))
    print("shape of words2id:",words2id.shape)
    print("id2word:\n",id2words.head(10))
    print("shape of id2words:",id2words.shape)


    #---------------------------------------生成word标注文件-----------------------------------------#
    f_train_in = open(file=inFile_train, encoding="utf-8")
    doc_words = ""
    doc_ids=""
    lines_train = f_train_in.readlines()
    for line_train in lines_train:
        #print("line:", line)
        line_train = line_train.strip()
        line_train_list = line_train.split(sep="\t")
        #print(line_list)
        if (line_train_list[0] != ""):
            id=words2id[line_train_list[0]]
            for i in range(int(line_train_list[2])):
                doc_words+=line_train_list[0][i]
                doc_words+=str(id)
                doc_ids+=(str(id)+" ")
        else:
            doc_words += "\n"
            doc_ids+="\n"
    f_train_out = open(file="word_train.txt", mode="w", encoding="utf-8")
    f_train_out.write(doc_words)
    f_train_out.close()

    f_train_out = open(file="word_train_tag.txt", mode="w", encoding="utf-8")
    f_train_out.write(doc_ids)
    f_train_out.close()


    f_valid_in = open(file=inFile_valid, encoding="utf-8")
    doc_words = ""
    doc_ids=""
    lines_valid = f_valid_in.readlines()
    for line_valid in lines_valid:
        #print("line:", line)
        line_valid = line_valid.strip()
        line_valid_list = line_valid.split(sep="\t")
        #print(line_list)
        if (line_valid_list[0] != ""):
            id=words2id[line_valid_list[0]]
            for i in range(int(line_valid_list[2])):
                doc_words+=line_valid_list[0][i]
                doc_words+=str(id)
                doc_ids+=(str(id)+" ")
        else:
            doc_words += "\n"
            doc_ids+="\n"
    f_valid_out = open(file="word_valid.txt", mode="w", encoding="utf-8")
    f_valid_out.write(doc_words)
    f_valid_out.close()

    f_valid_out = open(file="word_valid_tag.txt", mode="w", encoding="utf-8")
    f_valid_out.write(doc_ids)
    f_valid_out.close()

    f_test_in = open(file=inFile_test, encoding="utf-8")
    doc_words = ""
    doc_ids = ""
    lines_test = f_test_in.readlines()
    for line_test in lines_test:
        # print("line:", line)
        line_test = line_test.strip()
        line_test_list = line_test.split(sep="\t")
        # print(line_list)
        if (line_test_list[0] != ""):
            id = words2id[line_test_list[0]]
            for i in range(int(line_test_list[2])):
                doc_words += line_test_list[0][i]
                doc_words += str(id)
                doc_ids += (str(id) + " ")
        else:
            doc_words += "\n"
            doc_ids += "\n"
    f_test_out = open(file="word_test.txt", mode="w", encoding="utf-8")
    f_test_out.write(doc_words)
    f_test_out.close()

    f_test_out = open(file="word_test_tag.txt", mode="w", encoding="utf-8")
    f_test_out.write(doc_ids)
    f_test_out.close()


'''


def toPOS(file):
    pass


def merge(file1,file2,outFile):
    doc=""
    f1=open(file=file1,encoding="utf-8")
    lines_f1=f1.readlines()
    for line_f1 in lines_f1:
        doc+=line_f1
    f2=open(file=file2,encoding="utf-8")
    lines_f2 = f2.readlines()
    for line_f2 in lines_f2:
        doc += line_f2
    f3=open(file=outFile,mode="w",encoding="utf-8")
    f3.write(doc)
    f3.close()




if __name__ =="__main__":
    if not os.path.exists("./data/corpus"):
        os.mkdir("./data/corpus/")

    print("[1]-> Conver raw .utf-8 files to prosody tagged files")
    toProsody(inFile="./data/raw/prosody_test_tag.utf8",outFile="./data/corpus/prosody_test.txt")
    toProsody(inFile="./data/raw/prosody_train_tag.utf8", outFile="./data/corpus/prosody_train.txt")
    toProsody2(inFile="./data/raw/prosody_valid_tag.rst", outFile="./data/corpus/prosody_valid.txt")

    #toWords(inFile="prosody_test_tag.utf8",outFile="word_test.txt")
    #toWords(inFile_train="prosody_train_tag.utf8",
    #        inFile_valid="prosody_valid_tag.rst",
    #        inFile_test="prosody_test_tag.utf8")

    print("[2]->merge prosody_train and prosody_valid files")
    merge(
        file1="./data/corpus/prosody_train.txt",
        file2="data/corpus/prosody_test.txt",
        outFile="data/corpus/prosody.txt"
    )

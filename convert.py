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
        line=line.strip()
        line_list=line.split(sep="\t")
        if(line_list[0]!=""):
            doc+=(line_list[0]+"#"+line_list[7])
        else:
            doc+="\n"
    #print(doc)
    f_out=open(file=outFile,mode="w",encoding="utf-8")
    f_out.write(doc)
    f_out.close()


def merge(file1,file2,file3,outFile):
    doc=""
    f1=open(file=file1,encoding="utf-8")
    lines_f1=f1.readlines()
    for line_f1 in lines_f1:
        doc+=line_f1
    f2=open(file=file2,encoding="utf-8")
    lines_f2 = f2.readlines()
    for line_f2 in lines_f2:
        doc += line_f2
    f3 = open(file=file3, encoding="utf-8")
    lines_f3 = f3.readlines()
    for line_f3 in lines_f3:
        doc += line_f3
    f4=open(file=outFile,mode="w",encoding="utf-8")
    f4.write(doc)
    f4.close()




#word pos info
def toPos(inFile_train,inFile_valid,inFile_test):
    #---------------------------------------生成pos列表--------------------------------------------#
    f_train_in = open(file=inFile_train, encoding="utf-8")
    f_valid_in = open(file=inFile_valid, encoding="utf-8")
    f_test_in = open(file=inFile_test, encoding="utf-8")

    pos=[]
    #收集所有pos
    #train
    lines_train = f_train_in.readlines()
    for line_train in lines_train:
        line_train = line_train.strip()
        line_train_list = line_train.split(sep="\t")
        if (line_train_list[0] != ""):
            pos.append(line_train_list[1])
    f_train_in.close()
    #valid
    lines_valid = f_valid_in.readlines()
    for line_valid in lines_valid:
        line_valid = line_valid.strip()
        line_valid_list = line_valid.split(sep="\t")
        if (line_valid_list[0] != ""):
            pos.append(line_valid_list[1])
    f_valid_in.close()
    #test
    lines_test = f_test_in.readlines()
    for line_test in lines_test:
        line_test = line_test.strip()
        line_test_list = line_test.split(sep="\t")
        if (line_test_list[0] != ""):
            pos.append(line_test_list[1])
    f_test_in.close()

    #print(pos)
    print("origin len of pos:",len(pos))
    sr_all_pos = pd.Series(data=pos)               # 列表做成pandas的Series
    pos = (sr_all_pos.value_counts()).index        # pos列表.统计每个pos类型出现的频率,同时相当于去重复,得到字的集合(这里还是Serieas的index对象)
    print(pos)
    print("len of cleaned:",pos.shape)
    pos_id = range(1, len(pos) + 1)             # 字的id列表,从1开始，因为准备把0作为填充值

    # words以及对应的id组件
    df_pos_ids=pd.DataFrame(data={"pos": pos, "id": pos_id})
    df_pos_ids. to_csv(path_or_buf="./data/dataset/pos_ids.csv", index=False, encoding="utf_8")

    pos2id = pd.Series(data=df_pos_ids["id"].values, index=df_pos_ids["pos"].values)
    id2pos = pd.Series(data=df_pos_ids["pos"].values, index=df_pos_ids["id"].values)

    #print("pos2id:\n",pos2id.head(10))
    #print("shape of pos2id:",pos2id.shape)
    #print("id2pos:\n",id2pos.head(10))
    #print("shape of id2pos:",id2pos.shape)


    #---------------------------------------生成pos标注文件-----------------------------------------#
    #training corpus
    f_train_in = open(file=inFile_train, encoding="utf-8")
    doc_pos = ""
    doc_ids=""
    lines_train = f_train_in.readlines()
    for line_train in lines_train:
        line_train = line_train.strip()
        line_train_list = line_train.split(sep="\t")
        #print("line_train_list:",line_train_list)
        if (line_train_list[0] != ""):
            id=pos2id[line_train_list[1]]
            doc_pos+=(line_train_list[0]+"/"+str(id))
            doc_ids+=(str(id)+" ")
        else:
            doc_pos += "\n"
            doc_ids+="\n"
    #save 2 files
    #f_train_out = open(file="./data/dataset/pos_train.txt", mode="w", encoding="utf-8")
    #f_train_out.write(doc_pos)
    #f_train_out.close()

    f_train_out = open(file="./data/dataset/pos_train_tag.txt", mode="w", encoding="utf-8")
    f_train_out.write(doc_ids)
    f_train_out.close()

    # validing corpus
    f_valid_in = open(file=inFile_valid, encoding="utf-8")
    doc_pos = ""
    doc_ids = ""
    lines_valid = f_valid_in.readlines()
    for line_valid in lines_valid:
        line_valid = line_valid.strip()
        line_valid_list = line_valid.split(sep="\t")
        # print("line_valid_list:",line_valid_list)
        if (line_valid_list[0] != ""):
            id = pos2id[line_valid_list[1]]
            doc_pos += (line_valid_list[0] + "/" + str(id))
            doc_ids += (str(id) + " ")
        else:
            doc_pos += "\n"
            doc_ids += "\n"
    # save 2 files
    #f_valid_out = open(file="./data/dataset/pos_valid.txt", mode="w", encoding="utf-8")
    #f_valid_out.write(doc_pos)
    #f_valid_out.close()

    f_valid_out = open(file="./data/dataset/pos_valid_tag.txt", mode="w", encoding="utf-8")
    f_valid_out.write(doc_ids)
    f_valid_out.close()

    #test corpus
    f_test_in = open(file=inFile_test, encoding="utf-8")
    doc_pos = ""
    doc_ids = ""
    lines_test = f_test_in.readlines()
    for line_test in lines_test:
        line_test = line_test.strip()
        line_test_list = line_test.split(sep="\t")
        if (line_test_list[0] != ""):
            id = pos2id[line_test_list[1]]
            doc_pos += (line_test_list[0] + "/" + str(id))
            doc_ids += (str(id) + " ")
        else:
            doc_pos += "\n"
            doc_ids += "\n"
    #f_test_out = open(file="./data/dataset/pos_test.txt", mode="w", encoding="utf-8")
    #f_test_out.write(doc_pos)
    #f_test_out.close()

    f_test_out = open(file="./data/dataset/pos_test_tag.txt", mode="w", encoding="utf-8")
    f_test_out.write(doc_ids)
    f_test_out.close()


#word length info(每个词的长度)
def toWordLength(inFile_train,inFile_valid,inFile_test):
    # ---------------------------------------生成length标注文件-----------------------------------------#
    # training corpus
    f_train_in = open(file=inFile_train, encoding="utf-8")
    doc_length = ""
    doc_ids = ""
    lines_train = f_train_in.readlines()
    for line_train in lines_train:
        line_train = line_train.strip()
        line_train_list = line_train.split(sep="\t")
        # print("line_train_list:",line_train_list)
        if (line_train_list[0] != ""):
            doc_length += (line_train_list[0] + "/" +line_train_list[2])
            doc_ids += (line_train_list[2] + " ")
        else:
            doc_length += "\n"
            doc_ids += "\n"
    # save 2 files
    #f_train_out = open(file="./data/dataset/length_train.txt", mode="w", encoding="utf-8")
    #f_train_out.write(doc_length)
    #f_train_out.close()

    f_train_out = open(file="./data/dataset/length_train_tag.txt", mode="w", encoding="utf-8")
    f_train_out.write(doc_ids)
    f_train_out.close()

    # validing corpus
    f_valid_in = open(file=inFile_valid, encoding="utf-8")
    doc_length = ""
    doc_ids = ""
    lines_valid = f_valid_in.readlines()
    for line_valid in lines_valid:
        line_valid = line_valid.strip()
        line_valid_list = line_valid.split(sep="\t")
        # print("line_valid_list:",line_valid_list)
        if (line_valid_list[0] != ""):
            doc_length += (line_valid_list[0] + "/" + line_valid_list[2])
            doc_ids += (line_valid_list[2] + " ")
        else:
            doc_length += "\n"
            doc_ids += "\n"
    # save 2 files
    #f_valid_out = open(file="./data/dataset/length_valid.txt", mode="w", encoding="utf-8")
    #f_valid_out.write(doc_length)
    #f_valid_out.close()

    f_valid_out = open(file="./data/dataset/length_valid_tag.txt", mode="w", encoding="utf-8")
    f_valid_out.write(doc_ids)
    f_valid_out.close()

    # test corpus
    f_test_in = open(file=inFile_test, encoding="utf-8")
    doc_length = ""
    doc_ids = ""
    lines_test = f_test_in.readlines()
    for line_test in lines_test:
        line_test = line_test.strip()
        line_test_list = line_test.split(sep="\t")
        if (line_test_list[0] != ""):
            doc_length += (line_test_list[0] + "/" + line_test_list[2])
            doc_ids += (line_test_list[2] + " ")
        else:
            doc_length += "\n"
            doc_ids += "\n"
    #f_test_out = open(file="./data/dataset/length_test.txt", mode="w", encoding="utf-8")
    #f_test_out.write(doc_length)
    #f_test_out.close()

    f_test_out = open(file="./data/dataset/length_test_tag.txt", mode="w", encoding="utf-8")
    f_test_out.write(doc_ids)
    f_test_out.close()

#word position info
def toWordAccum(inFile_train,inFile_valid,inFile_test):
    # ---------------------------------------生成accum标注文件-----------------------------------------#
    # training corpus
    f_train_in = open(file=inFile_train, encoding="utf-8")
    doc_position = ""
    doc_ids = ""
    lines_train = f_train_in.readlines()
    for line_train in lines_train:
        line_train = line_train.strip()
        line_train_list = line_train.split(sep="\t")
        # print("line_train_list:",line_train_list)
        if (line_train_list[0] != ""):
            doc_position += (line_train_list[0] + "/" + line_train_list[4])
            doc_ids += (line_train_list[4] + " ")
        else:
            doc_position += "\n"
            doc_ids += "\n"
    # save 2 files
    f_train_out = open(file="./data/dataset/accum_train.txt", mode="w", encoding="utf-8")
    f_train_out.write(doc_position)
    f_train_out.close()

    f_train_out = open(file="./data/dataset/accum_train_tag.txt", mode="w", encoding="utf-8")
    f_train_out.write(doc_ids)
    f_train_out.close()

    # validing corpus
    f_valid_in = open(file=inFile_valid, encoding="utf-8")
    doc_position = ""
    doc_ids = ""
    lines_valid = f_valid_in.readlines()
    for line_valid in lines_valid:
        line_valid = line_valid.strip()
        line_valid_list = line_valid.split(sep="\t")
        # print("line_valid_list:",line_valid_list)
        if (line_valid_list[0] != ""):
            doc_position += (line_valid_list[0] + "/" + line_valid_list[4])
            doc_ids += (line_valid_list[4] + " ")
        else:
            doc_position += "\n"
            doc_ids += "\n"
    # save 2 files
    f_valid_out = open(file="./data/dataset/accum_valid.txt", mode="w", encoding="utf-8")
    f_valid_out.write(doc_position)
    f_valid_out.close()

    f_valid_out = open(file="./data/dataset/accum_valid_tag.txt", mode="w", encoding="utf-8")
    f_valid_out.write(doc_ids)
    f_valid_out.close()

    # test corpus
    f_test_in = open(file=inFile_test, encoding="utf-8")
    doc_position = ""
    doc_ids = ""
    lines_test = f_test_in.readlines()
    for line_test in lines_test:
        line_test = line_test.strip()
        line_test_list = line_test.split(sep="\t")
        if (line_test_list[0] != ""):
            doc_position += (line_test_list[0] + "/" + line_test_list[4])
            doc_ids += (line_test_list[4] + " ")
        else:
            doc_position += "\n"
            doc_ids += "\n"
    f_test_out = open(file="./data/dataset/accum_test.txt", mode="w", encoding="utf-8")
    f_test_out.write(doc_position)
    f_test_out.close()

    f_test_out = open(file="./data/dataset/accum_test_tag.txt", mode="w", encoding="utf-8")
    f_test_out.write(doc_ids)
    f_test_out.close()

def toWordAccumReverse(inFile_train,inFile_valid,inFile_test):
    # ---------------------------------------生成accum标注文件-----------------------------------------#
    # training corpus
    f_train_in = open(file=inFile_train, encoding="utf-8")
    doc_position = ""
    doc_ids = ""
    lines_train = f_train_in.readlines()
    for line_train in lines_train:
        line_train = line_train.strip()
        line_train_list = line_train.split(sep="\t")
        # print("line_train_list:",line_train_list)
        if (line_train_list[0] != ""):
            doc_position += (line_train_list[0] + "/" + line_train_list[5])
            doc_ids += (line_train_list[5] + " ")
        else:
            doc_position += "\n"
            doc_ids += "\n"
    # save 2 files
    #f_train_out = open(file="./data/dataset/accum_reverse_train.txt", mode="w", encoding="utf-8")
    #f_train_out.write(doc_position)
    #f_train_out.close()

    f_train_out = open(file="./data/dataset/accum_reverse_train_tag.txt", mode="w", encoding="utf-8")
    f_train_out.write(doc_ids)
    f_train_out.close()

    # validing corpus
    f_valid_in = open(file=inFile_valid, encoding="utf-8")
    doc_position = ""
    doc_ids = ""
    lines_valid = f_valid_in.readlines()
    for line_valid in lines_valid:
        line_valid = line_valid.strip()
        line_valid_list = line_valid.split(sep="\t")
        # print("line_valid_list:",line_valid_list)
        if (line_valid_list[0] != ""):
            doc_position += (line_valid_list[0] + "/" + line_valid_list[5])
            doc_ids += (line_valid_list[5] + " ")
        else:
            doc_position += "\n"
            doc_ids += "\n"
    # save 2 files
    #f_valid_out = open(file="./data/dataset/accum_reverse_valid.txt", mode="w", encoding="utf-8")
    #f_valid_out.write(doc_position)
    #f_valid_out.close()

    f_valid_out = open(file="./data/dataset/accum_reverse_valid_tag.txt", mode="w", encoding="utf-8")
    f_valid_out.write(doc_ids)
    f_valid_out.close()

    # test corpus
    f_test_in = open(file=inFile_test, encoding="utf-8")
    doc_position = ""
    doc_ids = ""
    lines_test = f_test_in.readlines()
    for line_test in lines_test:
        line_test = line_test.strip()
        line_test_list = line_test.split(sep="\t")
        if (line_test_list[0] != ""):
            doc_position += (line_test_list[0] + "/" + line_test_list[5])
            doc_ids += (line_test_list[5] + " ")
        else:
            doc_position += "\n"
            doc_ids += "\n"
    #f_test_out = open(file="./data/dataset/accum_reverse_test.txt", mode="w", encoding="utf-8")
    #f_test_out.write(doc_position)
    #f_test_out.close()

    f_test_out = open(file="./data/dataset/accum_reverse_test_tag.txt", mode="w", encoding="utf-8")
    f_test_out.write(doc_ids)
    f_test_out.close()

#word position info
def toWordPosition(inFile_train,inFile_valid,inFile_test):
    # ---------------------------------------生成position标注文件-----------------------------------------#
    # training corpus
    f_train_in = open(file=inFile_train, encoding="utf-8")
    doc_position = ""
    doc_ids = ""
    lines_train = f_train_in.readlines()
    i=1
    for line_train in lines_train:
        line_train = line_train.strip()
        line_train_list = line_train.split(sep="\t")
        # print("line_train_list:",line_train_list)
        if (line_train_list[0] != ""):
            doc_position += (line_train_list[0] + "/" + str(i))
            doc_ids += (str(i) + " ")
            i+=1
        else:
            doc_position += "\n"
            doc_ids += "\n"
            i=1
    # save 2 files
    #f_train_out = open(file="./data/dataset/position_train.txt", mode="w", encoding="utf-8")
    #f_train_out.write(doc_position)
    #f_train_out.close()

    f_train_out = open(file="./data/dataset/position_train_tag.txt", mode="w", encoding="utf-8")
    f_train_out.write(doc_ids)
    f_train_out.close()

    # validing corpus
    f_valid_in = open(file=inFile_valid, encoding="utf-8")
    doc_position = ""
    doc_ids = ""
    lines_valid = f_valid_in.readlines()
    i = 1
    for line_valid in lines_valid:
        line_valid = line_valid.strip()
        line_valid_list = line_valid.split(sep="\t")
        # print("line_valid_list:",line_valid_list)
        if (line_valid_list[0] != ""):
            doc_position += (line_valid_list[0] + "/" + str(i))
            doc_ids += (str(i) + " ")
            i += 1
        else:
            doc_position += "\n"
            doc_ids += "\n"
            i = 1
    # save 2 files
    #f_valid_out = open(file="./data/dataset/position_valid.txt", mode="w", encoding="utf-8")
    #f_valid_out.write(doc_position)
    #f_valid_out.close()

    f_valid_out = open(file="./data/dataset/position_valid_tag.txt", mode="w", encoding="utf-8")
    f_valid_out.write(doc_ids)
    f_valid_out.close()

    # test corpus
    f_test_in = open(file=inFile_test, encoding="utf-8")
    doc_position = ""
    doc_ids = ""
    lines_test = f_test_in.readlines()
    i=1
    for line_test in lines_test:
        line_test = line_test.strip()
        line_test_list = line_test.split(sep="\t")
        if (line_test_list[0] != ""):
            doc_position += (line_test_list[0] + "/" + str(i))
            doc_ids += (str(i) + " ")
            i+=1
        else:
            doc_position += "\n"
            doc_ids += "\n"
            i=1
    #f_test_out = open(file="./data/dataset/position_test.txt", mode="w", encoding="utf-8")
    #f_test_out.write(doc_position)
    #f_test_out.close()

    f_test_out = open(file="./data/dataset/position_test_tag.txt", mode="w", encoding="utf-8")
    f_test_out.write(doc_ids)
    f_test_out.close()




if __name__ =="__main__":
    if not os.path.exists("./data/corpus"):
        os.mkdir("./data/corpus/")
    if not os.path.exists("./data/dataset"):
            os.mkdir("./data/dataset/")
    if not os.path.exists("./result"):
        os.mkdir("./result")

    print("[1]-> Conver raw .utf-8 files to prosody tagged files")
    toProsody(inFile="./data/raw/prosody_test_tag.utf8",outFile="./data/corpus/prosody_test.txt")
    toProsody(inFile="./data/raw/prosody_train_tag.utf8", outFile="./data/corpus/prosody_train.txt")
    toProsody(inFile="./data/raw/prosody_valid_tag.utf8", outFile="./data/corpus/prosody_valid.txt")


    print("[2]->merge prosody_train and prosody_valid and prosody_test files")
    merge(
        file1="./data/corpus/prosody_train.txt",
        file2="data/corpus/prosody_valid.txt",
        file3="data/corpus/prosody_test.txt",
        outFile="data/corpus/prosody.txt"
    )


    print("[3]->generate pos files")
    toPos(inFile_train="./data/raw/prosody_train_tag.utf8",
            inFile_valid="./data/raw/prosody_valid_tag.utf8",
            inFile_test="./data/raw/prosody_test_tag.utf8"
    )

    print("[4]->generate length files")
    toWordLength(inFile_train="./data/raw/prosody_train_tag.utf8",
            inFile_valid="./data/raw/prosody_valid_tag.utf8",
            inFile_test="./data/raw/prosody_test_tag.utf8"
     )

    print("[5]->generate accmulate files")
    toWordAccum(inFile_train="./data/raw/prosody_train_tag.utf8",
                inFile_valid="./data/raw/prosody_valid_tag.utf8",
                inFile_test="./data/raw/prosody_test_tag.utf8"
            )

    print("[6]->generate accmulate reverse files")
    toWordAccumReverse(inFile_train="./data/raw/prosody_train_tag.utf8",
                       inFile_valid="./data/raw/prosody_valid_tag.utf8",
                       inFile_test="./data/raw/prosody_test_tag.utf8"
                    )

    print("[7]->generate position files")
    toWordPosition(inFile_train="./data/raw/prosody_train_tag.utf8",
                   inFile_valid="./data/raw/prosody_valid_tag.utf8",
                   inFile_test="./data/raw/prosody_test_tag.utf8"
                )


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readData(file):
    list=[]
    f=open(file=file)
    lines=f.readlines()
    for line in lines:
        line=line.strip()
        list.append(float(line))
    return list


if __name__ =="__main__":
    #plt.xlabel("Mini-Batch")
    #plt.ylabel("Accuracy")
    #list=readData(file="train_accuracy_epoch1.txt")
    #plt.plot(list,"r")
    list2=readData(file="train_loss_epoch1.txt")
    plt.xlabel("Mini-Batch")
    plt.ylabel("Loss")
    plt.plot(list2,"r")
    plt.show()
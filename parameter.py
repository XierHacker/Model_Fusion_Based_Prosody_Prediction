#basic architecture
CHAR_EMBEDDING_SIZE=128                 #字嵌入维度
WORD_EMBEDDING_SIZE=128                 #词嵌入维度
INPUT_SIZE=WORD_EMBEDDING_SIZE           #词嵌入维度

MAX_EPOCH=3                            #最大迭代次数
LAYER_NUM=2                             #lstm层数2
HIDDEN_UNITS_NUM=256                    #隐藏层结点数量
HIDDEN_UNITS_NUM2=256                   #隐藏层2结点数量
BATCH_SIZE=20                           #batch大小

#learning rate
LEARNING_RATE=0.003                      #学习率
DECAY=0.05                               #衰减系数

#Weaken Overfitting
KEEP_PROB=0.5                             #全连接 dropout 比率
INPUT_KEEP_PROB=1.0                       #rnn input  dropout比率
OUTPUT_KEEP_PROB=0.5                      #rnn output dropout 比率
LAMBDA_PW=0.001                           #PW层级正则化系数
LAMBDA_PPH=0.001                          #PW层级正则化系数
LAMBDA_IPH=0.005                          #PW层级正则化系数


#can't modify
CLASS_NUM=2                             #类别数量2(N,B)
POS_NUM=37                              #词性信息数量
LENGTH_NUM=8                            #长度信息数量
MAX_SENTENCE_SIZE=28                    #固定句子长度为28 (从整个数据集得来)
TIMESTEP_SIZE=MAX_SENTENCE_SIZE         #LSTM的time_step应该和句子长度一致
WORD_VOCAB_SIZE=393256                   # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到

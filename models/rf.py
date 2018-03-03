import numpy as np
from sklearn.preprocessing import OneHotEncoder

#a=[[1],[2],[3],[4]]
#encoder=OneHotEncoder()
#encoder.fit(a)
#b=encoder.transform(a).toarray()

#print(b)

a=np.array([1,2,5,5])
def onehot(array):
    max_value=np.max(array)
    a=np.zeros(shape=(array.shape[0],max_value),dtype=np.int32)
    for i in range(array.shape[0]):
        a[i,array[i]-1]=1
    return a


b=onehot(a)
print(a)
print(b)
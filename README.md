## **Requirements**
>**python3.5+**

>**tensorflow>=1.6**

>**numpy**

>**pandas**

>**scikit-learn**

>**gensim**


## steps
### **----------------------data processing-----------------------**
#### 1.run `python convert.py`
>trans .utf-8 raw files to prosody tagged files

#### 2.run `python data_processing.py`
>trans prosody tagged files to dataset

### **-------------------use models to prediction-----------------**
#### `cd models`
>into models

#### run `python bilstm_cbow.py`
>use bilstm_cbow to do prosody prediction 


#### run `python alignment.py`
>use alignment to do prosody prediction 
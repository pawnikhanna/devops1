#!/usr/bin/env python
# coding: utf-8

# In[41]:


import tensorflow as tf 
from tensorflow import keras
from sklearn import preprocessing, model_selection
import numpy as np 
import pandas as pd 


# In[42]:


df = pd.read_csv('6 class csv.csv')
df.head()


# In[43]:


df.isnull().values.any()


# In[44]:


df.corr()


# In[45]:


import matplotlib.pyplot as plt

def plot_corr(df):
    corr = df.corr()
    fig,ax = plt.subplots(figsize = (6,6))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)),corr.columns)
    plt.yticks(range(len(corr.columns)),corr.columns)
    
    
plot_corr(df)


# In[46]:


df.info()


# In[47]:


x = np.array(df.drop(['Star type', 'Star color','Spectral Class'],1))   


# In[48]:


y = np.array(df['Star type'], dtype ='float')
y.shape = (len(y),1)


# In[49]:


x_train ,x_test , y_train, y_test = model_selection.train_test_split(x,y, test_size = 0.3) 


# In[50]:


x_f_train = preprocessing.scale(x_train)
x_f_test = preprocessing.scale(x_test)
y_f_train = y_train
y_f_test = y_test


# In[51]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())


# In[52]:


x_f_train.shape


# In[53]:


y_f_train.shape


# In[54]:


from keras.layers import Dense


# In[55]:


model.add(tf.keras.layers.Dense(200,activation ='relu'))


# In[56]:


model.add(tf.keras.layers.Dense(300,activation = 'relu'))


# In[57]:


model.add(tf.keras.layers.Dense(6,activation = 'softmax'))


# In[58]:


from keras.optimizers import RMSprop


# In[59]:


model.compile(optimizer = 'Adam',
       loss = 'sparse_categorical_crossentropy',
       metrics=['accuracy'])


# In[60]:


model.fit(x_f_train,y_f_train, epochs = 100)


# In[61]:


val_loss,val_acc = model.evaluate(x_f_test,y_f_test)
print("Loss % = {} , Accuracy % = {} ".format(val_loss*100,val_acc*100))


# In[62]:


model.save('star_classification.h5')


# In[ ]:





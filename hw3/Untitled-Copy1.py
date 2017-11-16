
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"


# In[4]:


# %load train.py
import numpy as np
import pandas as pd
import os
import sys
import time
import keras
import argparse
from six.moves import cPickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from utils import DataLoader, DataGenerator


# In[5]:


# %load model.py
"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import print_function
from __future__ import absolute_import

import keras

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import ZeroPadding2D
from keras import backend as K
#congrats! the last year!!!!! love you waiting for you 


# In[6]:


d = DataLoader(train_file="train.csv",num_classes=7)
X_train, X_val, y_train, y_val = train_test_split(d.X_train, d.Y_train, test_size=0.25)


# In[70]:


datagen = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             fill_mode='nearest')

train_generator_aug = datagen.flow(X_train,y_train)


# In[49]:


num_classes = 7
img_input = Input(shape=(48,48,1))
# Block 1
x = Conv2D(64, (5, 5), padding='valid', name='block1_conv')(img_input)
x = Activation('relu')(x)
x = ZeroPadding2D(padding=(2,2))(x)
x = MaxPooling2D(pool_size=(5,5),strides=(2,2))(x)
x = ZeroPadding2D(padding=(1,1))(x)

x = Conv2D(128, (3, 3), padding='same', name='block2_conv')(x)
x = Activation('relu')(x)
x = ZeroPadding2D(padding=(1,1))(x)
x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)

x = Conv2D(256, (3, 3), padding='same', name='block3_conv')(x)
x = Activation('relu')(x)
x = ZeroPadding2D(padding=(1,1))(x)
x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)

x = Conv2D(256, (3, 3), padding='same', name='block4_conv')(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(1, 1))(x)

x = Flatten(name='flatten')(x)
x = Dense(2048,activation='relu',name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(2048,activation='relu',name='fc2')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax', name='predictions')(x)

md = Model(img_input, x, name='mymodel')
md.summary()


# In[50]:


learning_rate = 0.0005
batch_size = 30
n_epoch = 100
save_dir = 'save/'


# In[79]:


md = keras.models.load_model(os.path.join(save_dir,'model_t83.h5'))


# In[80]:


# opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt = keras.optimizers.Adadelta(lr=0.001, rho=0.95, epsilon=1e-08, decay=0.0)

md.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


# In[81]:


checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(save_dir,'model_t84.h5'),monitor='val_loss',verbose=1,save_best_only=True,mode='auto')
csv_logger = keras.callbacks.CSVLogger(os.path.join(save_dir,'training_t84.log'))


# In[82]:


md.fit_generator(#generator = train_generator.generate(batch_size=batch_size),
                generator = train_generator_aug,
                epochs=n_epoch,
                steps_per_epoch = 400,
                validation_data = (X_val,y_val),
                callbacks=[checkpoint,csv_logger])


# In[ ]:





# In[41]:


md.save(os.path.join(save_dir,'model_ovf.h5'))


# In[27]:


dt = DataLoader(test_file="test.csv",num_classes=num_classes)


# In[35]:


best_md = keras.models.load_model(os.path.join(save_dir,'model_t4.h5'))


# In[36]:


y_pred = best_md.predict(dt.X_test)
labels = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
output = pd.DataFrame({"id":range(len(y_pred)),"label":labels})
output.to_csv("output_t4.csv", index=False)


# In[46]:


y_pred = md.predict(dt.X_test)
labels = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
output = pd.DataFrame({"id":range(len(y_pred)),"label":labels})
output.to_csv("output_ovf.csv", index=False)


# In[ ]:





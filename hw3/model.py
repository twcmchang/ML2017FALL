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
def model(input_shape=None, num_classes=7):
    img_input = Input(shape=input_shape)
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

    model = Model(img_input, x, name='mymodel')
    return model

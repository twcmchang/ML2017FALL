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
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras import backend as K
#congrats! the last year!!!!! love you waiting for you 
def model(input_shape=None, num_classes=7):
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), padding='valid', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2, 2), name='block1_pool1')(x)
    x = Conv2D(64, (3, 3), padding='valid', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool2')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='valid', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2, 2), name='block2_pool1')(x)
    x = Conv2D(128, (3, 3), padding='valid', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2),name='block2_pool2')(x)

    # # Block 3
    x = Conv2D(256, (3, 3), padding='valid', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2, 2), name='block3_pool1')(x)
    x = Conv2D(256, (3, 3), padding='valid', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2, 2), name='block3_pool2')(x)
    x = Conv2D(256, (3, 3), padding='valid', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2, 2), name='block3_pool3')(x)

    # # Block 4
    # x = Conv2D(512, (3, 3), padding='valid', name='block4_conv1')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Conv2D(512, (3, 3), padding='valid', name='block4_conv2')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Conv2D(512, (3, 3), padding='valid', name='block4_conv3')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # # Block 5
    # x = Conv2D(512, (3, 3), padding='valid', name='block5_conv1')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Conv2D(512, (3, 3), padding='valid', name='block5_conv2')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Conv2D(512, (3, 3), padding='valid', name='block5_conv3')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    #x = Flatten(name='flatten')(x)
    x = Dense(1024,activation='relu',name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024,activation='relu',name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='mymodel')
    return model

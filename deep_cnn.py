
import tensorflow as tf
tf.keras.utils.set_random_seed(0)
import numpy as np

from absl import app
from absl import flags

from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import MaxPool2D, Activation, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, auc, roc_curve

import os

# Annealer
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def my_sparse_categorical_crossentropy(y_true, y_pred):
    return tf.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True, reduction=tf.losses.Reduction.NONE)

def target_model_fn():

    #base_model_1 = tf.keras.applications.VGG16(include_top=False,weights='imagenet',input_shape=(32,3,3),classes=10)
    base_model_1 = tf.keras.applications.VGG16(include_top=False,weights='imagenet',input_shape=(32,32,3),classes=10)
    model_1= Sequential()
    model_1.add(base_model_1) #Adds the base model (in this case vgg19 to model_1)
    model_1.add(Flatten())

    model_1.add(Dense(1024,activation=('tanh'),input_dim=512))

    model_1.add(Dense(512,activation=('tanh'))) 
    model_1.add(Dense(256,activation=('tanh'))) 

    model_1.add(Dense(128,activation=('tanh')))

    model_1.add(Dense(10,activation=('softmax'))) 
    learn_rate=.001
    #sgd=SGD(lr=learn_rate,momentum=.9,nesterov=False)
    sgd=SGD(lr=learn_rate,momentum=.9,nesterov=False,decay=1e-6)
    
    #parallel_model = multi_gpu_model(model_1, gpus=4) 
    model_1.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
    return model_1
def target_model_fn20():
    
    base_model_1 = tf.keras.applications.VGG16(include_top=False,weights='imagenet',input_shape=(32,32,3),classes=10)
    model_1= Sequential()
    model_1.add(base_model_1) #Adds the base model (in this case vgg19 to model_1)
    model_1.add(Flatten())
    
    model_1.add(Dense(1024,activation=('tanh'),input_dim=512))
 
    model_1.add(Dense(512,activation=('tanh'))) 

    model_1.add(Dense(256,activation=('tanh'))) 

    model_1.add(Dense(128,activation=('tanh')))
    model_1.add(Dense(10,activation=('softmax'))) 
    learn_rate=.001
  
  
    sgd=SGD(lr=learn_rate,momentum=.9,nesterov=False)
    
    #parallel_model = multi_gpu_model(model_1, gpus=4)my_sparse_categorical_crossentropy
    model_1.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #model_1.compile(optimizer='sgd',loss='KLDivergence',metrics=['accuracy'])
    return model_1


def target_model_fn5():
    model = Sequential()
    model.add(Flatten(input_shape=(600,  )))
    model.add(Dense(1024, activation='sigmoid'))
    
    model.add(Dense(512, activation='sigmoid'))

    model.add(Dense(256, activation='sigmoid'))
  
    model.add(Dense(128, activation='sigmoid'))
    
    model.add(Dense(100, activation='softmax'))
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model
def target_model_fn50():
  
    model = Sequential()
    model.add(Flatten(input_shape=(600,  )))
    model.add(Dense(1024, activation='sigmoid'))
    
    model.add(Dense(512, activation='sigmoid'))

    model.add(Dense(256, activation='sigmoid'))
  
    model.add(Dense(128, activation='sigmoid'))
    
    model.add(Dense(100, activation='softmax'))
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.001,), loss="categorical_crossentropy", metrics=["accuracy"])

    return model
  
def target_model_fn51():

    model = Sequential()
    model.add(Flatten(input_shape=(600,  )))
    model.add(Dense(1024, activation='sigmoid'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(100, activation='softmax'))
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model
def target_model_fn777V3():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, input_shape=(32, 32, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    #model.add(Dropout(0.3))

    model.add(Conv2D(128, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    #model.add(Dropout(0.3))

    model.add(Conv2D(256, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    #model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.3))

    model.add(Dense(10, activation='softmax'))
  	# compile model
    opt = SGD(learning_rate=0.001,)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def attack_model_fn():
  
    model = tf.keras.models.Sequential()


    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dense(2, activation="softmax"))
    model.compile("sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

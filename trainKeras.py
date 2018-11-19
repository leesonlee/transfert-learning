import tensorflow as tf
import numpy as np
import keras
import h5py
from PIL import Image
from keras import backend as K
# from keras.models import Sequential
# from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
# from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
# from keras.layers.convolutional import *
# from sklearn.metrics import confusion_matrix
from keras import Model
# from keras.applications.imagenet_utils import preprocess_input

# import itertools
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

# specifying the path to our data:
train_path = "D:\\source\\data\\Keras-transfert-learning\\train"
valid_path = "D:\\source\\data\\Keras-transfert-learning\\valid"
test_path = "D:\\source\\data\\Keras-transfert-learning\\test"

# loading in the data in the specific batches:
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=["contrat","","FEsignatureOK","FEsignatureKO","signatureOK"],batch_size=36)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=["contrat","","FEsignatureOK","FEsignatureKO","signatureOK"],batch_size=13)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=["contrat","","FEsignatureOK","FEsignatureKO","signatureOK"],batch_size=10)

# get base model
base_model = keras.applications.VGG16(weights='imagenet', include_top=True)

# make a reference to VGG's input layer
inp = base_model.input

# make a new softmax layer with num of outputs
new_classification_layer = Dense(5, activation='sigmoid')

# connect our new layer to the second to last layer in VGG, and make a reference to it
out = new_classification_layer(base_model.layers[-2].output)

# create a new network between inp and out
model_new = Model(inp, out)
model_new.summary()

# make all layers untrainable by freezing weights (except for last layer)
for layer in model_new.layers[:-1]:
    layer.trainable = False

    # ensure the last layer is trainable/not frozen    
for layer in model_new.layers[-1:]:
    layer.trainable = True

# check to see it is correct:
for layer in model_new.layers:
    print(layer.trainable)

model_new.compile(Adam(lr=.0001), loss="categorical_crossentropy", metrics=['accuracy'])
model_new.summary()

# We train the model, in this case using a low size of epochs, since the model learns the data quite quickly
model_new.fit_generator(train_batches, steps_per_epoch=101,
                   validation_data=valid_batches, validation_steps=106, epochs=25, verbose=2)

model_new.fit_generator(train_batches, steps_per_epoch=101,
                   validation_data=valid_batches, validation_steps=106, epochs=5, verbose=2)

model_new.save("VGG16signature.h5")

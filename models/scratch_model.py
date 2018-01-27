##########################################################
######### First Model for steganalysis based on CNN #######
##########################################################


"""
info:
Dropout basically shuts down a few weights of the network randomly.
"""

# import libraries
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.utils import np_utils
from keras.layers import Merge, Lambda, Layer, GlobalAveragePooling2D
from keras.models import Sequential, InputLayer, Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

class ScratchModel(object):
    """docstring for ScratchModel"""

    def __init__(self, input_shape, nbr_classes):
        super(ScratchModel, self).__init__()
        self.nbr_classes = nbr_classes
        self.input_shape = input_shape

    # Defining the model
    def get_model_architecture(self):

        model = Sequential()
       
        # Input Layer
        model.add(InputLayer(input_shape=self.input_shape))

        # Hidden Layer
        model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1,
                         padding="same", use_bias=False, trainable=True))
        
        # Classification
        # this converts our 2D feature maps to 1D feature vectors
        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model

    # compile the model
    def compile_model(self, model, loss, optimizer, metrics):
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

##########################################################
######### First Model for steganalysis based on CNN #######
##########################################################


"""
info:
Dropout basically shuts down a few weights of the network randomly.
"""

# import libraries

import numpy as np
from keras import backend as K
K.set_image_dim_ordering('tf')


from keras.utils import np_utils
from keras.layers import Merge, Lambda, Layer, GlobalAveragePooling2D
from keras.models import Sequential, InputLayer, Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
import numpy



class ScratchModel(object):
    """docstring for ScratchModel"""

    def __init__(self, input_shape, nbr_classes):
        super(ScratchModel, self).__init__()
        self.nbr_classes = nbr_classes
        self.input_shape = input_shape

    # Defining the model
    def get_model_architecture(self):

        KV = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=np.float32)/12
        K1 = np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32)/4
        K2 = np.array([[2,-1,2],[-1,-4,-1],[2,-1,2]], dtype=np.float32)/4

        KM = np.array([[0,0,5.2,0,0],[0,23.4,36.4,23.4,0],[5.2,36.4,-261,36.4,5.2],[0,23.4,36.4,23.4,0],[0,0,5.2,0,0]], dtype=np.float32)/261
        GH = np.array([[0.0562,-0.1354,0,0.1354,-0.0562],[0.0818,-0.1970,0,0.1970,-0.0818],[0.0926,-0.2233,0,0.2233,-0.0926],[0.0818,-0.1970,0,0.1970,-0.0818],[0.0562,-0.1354,0,0.1354,-0.0562]], dtype=np.float32)
        GV = np.fliplr(GH).T.copy() 

        F1 = numpy.reshape(KV, (KV.shape[0],KV.shape[1],1,1))
        F2 = numpy.reshape(KM, (KM.shape[0],KM.shape[1],1,1))
        F3 = numpy.reshape(GH, (GH.shape[0],GH.shape[1],1,1))
        F4 = numpy.reshape(GV, (GV.shape[0],GV.shape[1],1,1))
        bias=numpy.array([0])


        model = Sequential()




        model.add(InputLayer(input_shape=self.input_shape))

        
        model.add(Conv2D(1, (5,5), padding="same", bias=False))

        model.add(Conv2D(8, (5, 5), padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(16, (5, 5), padding='same'))
        model.add(Activation('relu'))
        
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2)))

       

        # this converts our 3D feature maps to 1D feature vectors
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        

        model.add(Dense(2))
        model.add(Activation('sigmoid'))

        return model

    def compile_model(self, model, loss, optimizer, metrics):
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

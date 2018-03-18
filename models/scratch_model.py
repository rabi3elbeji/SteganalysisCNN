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
from keras.layers.advanced_activations import LeakyReLU, PReLU

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
        model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=2,
                         padding="same", use_bias=False, trainable=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.1))


        model.add(Conv2D(filters=5, kernel_size=(5, 5), strides=2,
                         padding="same", use_bias=True, trainable=True))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.1))
        model.add(Dropout(0.1))
        
     
        model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=2,
                         padding="same", use_bias=True, trainable=True))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.1))
        model.add(Dropout(0.1))
 
     

        # Classification
        # this converts our 2D feature maps to 1D feature vectors
        model.add(Flatten())
        model.add(Dense(200))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.1))
        model.add(Dense(200))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.1))
        model.add(Dropout(0.5))

        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model

        return model

    def compile_model(self, model, loss, optimizer, metrics):
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

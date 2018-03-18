###############################################################################
############### Class for view model weights and layers class #################
###############################################################################

# import libraries
import os
import cv2
import numpy as np
import json
from collections import namedtuple
from keras.models import model_from_json
from utils.plot_data import PlotData
from keras import backend as K
K.set_image_dim_ordering('tf')

# model type
model_type = 'wow_128_01'
input_image_path = './test.jpg'


# path to saved model files
saved_model_weights_path = './trained_for_pred/' + \
    model_type + '/model/Best-weights.h5'
saved_model_arch_path = './trained_for_pred/' + \
    model_type + '/model/scratch_model.json'


# ==> outputs >> weigths path
saved_weights_conv1_path = './trained_for_pred/' + \
    model_type + '/details/weigts_conv1.png'
saved_weights_conv2_path = './trained_for_pred/' + \
    model_type + '/details/weigts_conv2.png'
saved_weights_conv3_path = './trained_for_pred/' + \
    model_type + '/details/weigts_conv3.png'
saved_hpf_path = './trained_for_pred/' + \
    model_type + '/details/hpf_filter.png'


saved_original_image_path = './trained_for_pred/' + \
    model_type + '/details/original_image.png'
saved_hpf_image_path = './trained_for_pred/' + \
    model_type + '/details/hpf_image.png'
saved_images_conv1_path = './trained_for_pred/' + \
    model_type + '/details/images_conv1.png'
saved_images_conv2_path = './trained_for_pred/' + \
    model_type + '/details/images_conv2.png'
saved_images_conv3_path = './trained_for_pred/' + \
    model_type + '/details/images_conv3.png'


saved_images_pool1_path = './trained_for_pred/' + \
    model_type + '/details/images_pool1.png'
saved_images_pool2_path = './trained_for_pred/' + \
    model_type + '/details/images_pool2.png'
saved_images_pool3_path = './trained_for_pred/' + \
    model_type + '/details/images_pool3.png'

# layer number that wich to extract weights
num_layer = 1  # 0 is the input layer
# neuron number
num_neuron = 0

# HPF filter
# HPF kernel
hpf_kernel = np.array(
    [[-1,  2,  -2,  2, -1],
     [2, -6,   8, -6,  2],
     [-2,  8, -12,  8, -2],
     [2, -6,   8, -6,  2],
     [-1,  2,  -2,  2, -1]])


# Load architecture and weights of the model
def load_model():
    print("===================== load model =========================")
    json_file = open(saved_model_arch_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(saved_model_weights_path)
    print("Loaded model from disk")
    return loaded_model


def preprocess(image):
    image = cv2.imread(image, 0)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(saved_original_image_path, image)
    # app the HPF filter
    image = cv2.filter2D(image, -1, hpf_kernel)
    cv2.imwrite(saved_hpf_image_path, image)
    image = np.array(image)
    image = np.expand_dims(image, axis=2)

    image = image.astype('float32')
    image /= 255

    return image


def main():
    # init class PlotData
    plotData = PlotData()
    # load the model
    model = load_model()
    layer_input = model.layers[0]
    layer_conv1 = model.layers[1]
    weights_conv1 = layer_conv1.get_weights()[0]
 
    

    input_image = preprocess(input_image_path)

    output_conv1 = K.function(inputs=[layer_input.input],
                              outputs=[layer_conv1.output])

    layer_output1 = output_conv1([[input_image]])[0]

    plotData.plot_conv_output_1_4(layer_output1, saved_images_conv1_path)

    image = cv2.imread(input_image_path, 0)
    plotData.plot_conv_weights_3_4(image, weights_conv1, layer_output1, saved_weights_conv1_path)



    '''
    plotData.plot_conv_output_8_4(pool_output1, saved_images_pool1_path)
    plotData.plot_conv_output_8_4(pool_output2, saved_images_pool2_path)
    plotData.plot_conv_output(pool_output3, saved_images_pool3_path)
    '''
    
    '''
    layer_output2 = output_conv2([[input_image]])[0]
    plotData.plot_conv_output(layer_output2, saved_images_conv2_path)
    layer_output3 = output_conv3([[input_image]])[0]
    plotData.plot_conv_output(layer_output3, saved_images_conv3_path)
    '''
    #layer_conv2 = model.layers[4]
    #layer_conv2 = model.layers[3]

    #weights_conv2 = layer_conv2.get_weights()[0]
    #weights_conv3 = layer_conv3.get_weights()[0]

    # print(weights_conv3.shape)
    #layer_conv2 = model.layers[3]

    #plotData.plot_conv_weights(weights_conv2, saved_weights_conv2_path)
    #plotData.plot_conv_weights(weights_conv3, saved_weights_conv3_path)
    pass

if __name__ == "__main__":
    main()

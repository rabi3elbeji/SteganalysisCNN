###################################################################
####################  Class for prediction ########################
###################################################################

# import libraries
import os
import time
import cv2
import numpy as np
import json
from collections import namedtuple
from keras.models import model_from_json


from keras import backend as K
K.set_image_dim_ordering('tf')

# model type
model_type = 'wow_128_10'

# path to saved model files
saved_model_weights_path = './trained_for_pred/'+model_type+'/model/Best-weights.h5'
saved_model_arch_path = './trained_for_pred/'+model_type+'/model/scratch_model.json'
saved_model_classid_path = './trained_for_pred/'+model_type+'/model/scratch_model_classid.json'

# model optimizer
model_optimizer_rmsprop = 'rmsprop'

# model loss function
model_loss_function = 'binary_crossentropy'

# model metrics
metrics = ['accuracy']

# set the input and the output paths
input_samples_stego_path = './datasets/dataset_'+model_type+'/test/stego/'
input_samples_cover_path = './datasets/dataset_'+model_type+'/test/cover/'
output_samples_path = './trained_for_pred/'+model_type+'/predictions/outputs/'
stego_class = 'stego'
cover_class = 'cover'

# variables for drawing
word_xmin, word_ymin, word_xmax, word_ymax = 0, 0, 60, 20
TEXT_FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_FONT_SCALE = 1
TEXT_THICKNESS = 2



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

# compile loaded model


def compile_model(model):
    model.compile(loss=model_loss_function,
                  optimizer=model_optimizer_rmsprop, metrics=metrics)
    return model

# Read image with some processing


def preprocess(image):
    image = cv2.imread(image, 0)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)

    image_cv = image.copy()
    if len(image_cv.shape) == 2:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)

    image = np.array(image)
    image = np.expand_dims(image, axis=2)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image /= 255

    return image_cv, image


def load_classid():
    class_by_id = {}
    with open(saved_model_classid_path) as json_data:
        # Parse JSON into an object with attributes corresponding to dict keys.
        d = json.load(json_data)
        for key, value in d.iteritems():
            class_by_id[value] = str(key)
    print(class_by_id)
    return class_by_id


def main():
    # load model
    model = load_model()
    # get model class id
    classid = load_classid()
    print("Load images from the input dir ...")
    # get images from input dirs
    input_stego_images = os.listdir(input_samples_stego_path)
    input_cover_images = os.listdir(input_samples_cover_path)

    # stats variable if missclassified number
    total_cover = 0
    total_stego = 0
    miss_cover = 0
    miss_stego = 0

    # loop stego dir
    for image_name in input_stego_images:
        # count stego img 
        total_stego += 1
        # Get the image
        image_path = input_samples_stego_path + image_name
        start_processing = time.time()
        # Resize + HPF
        image_cv, image = preprocess(image_path)
        # Prediction
        img_class = model.predict_classes(image)
        end_processing = time.time()
        print("End prdiction for image: " + image_name +
              ", elapsed time = " + str(end_processing - start_processing) + " s")
        # For drawing ..
        class_name = classid[img_class[0]]
        print(image_name, class_name)
        if class_name == stego_class:
            cv2.rectangle(image_cv, (word_xmin, word_ymin),
                          (word_xmax, word_ymax), (0, 255, 0), -1)
        else:
            # count miss stego
            miss_stego += 1
            cv2.rectangle(image_cv, (word_xmin, word_ymin),
                          (word_xmax, word_ymax), (0, 0, 255), -1)
        cv2.putText(image_cv, class_name, (5, 15), TEXT_FONT,
                    TEXT_FONT_SCALE, (0, 0, 0), TEXT_THICKNESS)
        cv2.imwrite(output_samples_path + 'stego_' +
                    image_name[:-3] + 'jpg', image_cv)

    # loop stego dir
    for image_name in input_cover_images:
        #count cover img
        total_cover += 1
        # Get the image
        image_path = input_samples_cover_path + image_name
        start_processing = time.time()
        # Resize + HPF
        image_cv, image = preprocess(image_path)
        # Prediction
        img_class = model.predict_classes(image)
        end_processing = time.time()
        print("End prdiction for image: " + image_name +
              ", elapsed time = " + str(end_processing - start_processing) + " s")
        # For drawing ..
        class_name = classid[img_class[0]]
        print(image_name, class_name)
        if class_name == cover_class:
            cv2.rectangle(image_cv, (word_xmin, word_ymin),
                          (word_xmax, word_ymax), (0, 255, 0), -1)
        else:
            #count miss cover
            miss_cover += 1
            cv2.rectangle(image_cv, (word_xmin, word_ymin),
                          (word_xmax, word_ymax), (0, 0, 255), -1)
        cv2.putText(image_cv, class_name, (5, 15), TEXT_FONT,
                    TEXT_FONT_SCALE, (0, 0, 0), TEXT_THICKNESS)
        cv2.imwrite(output_samples_path + 'cover_' +
                    image_name[:-3] + 'jpg', image_cv)

    print('Total cover images: '+str(total_cover)+" - miss cover number: "+str(miss_cover))
    print('Total stego images: '+str(total_stego)+" - miss stego number: "+str(miss_stego))

if __name__ == "__main__":
    main()

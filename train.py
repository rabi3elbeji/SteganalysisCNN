############################################################################
####################  The training script  #################################
############################################################################

# import of libraries
import os
from time import time
import numpy as np
import json
from models.scratch_model import ScratchModel
from utils.data_manager import DataManager
from utils.plot_data import PlotData
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import plot_model
from keras import callbacks
from keras.models import model_from_json
from keras.callbacks import TensorBoard
# Use tensorflow as backend for keras
from keras import backend as K
K.set_image_dim_ordering('tf')


# Model type
model_type = 'mg_128_01'

# test dataset img
model_dataset = 'dataset_' + model_type

# Dataset dir paths
train_data_dir = './datasets/' + model_dataset + '/train'
validation_data_dir = './datasets/' + model_dataset + '/validation'


# Images width, height, channels
img_height = 128
img_width = 128
num_channels = 1

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
image_shape = (img_height, img_width, num_channels)

# Class Number
class_number = 2

# model ==> output paths
model_png = './trained_for_pred/' + model_type + '/model/scratch_model.png'
model_summary_file = './trained_for_pred/' + \
    model_type + '/model/scratch_model_summary.txt'
saved_model_arch_path = './trained_for_pred/' + \
    model_type + '/model/scratch_model.json'
saved_model_classid_path = './trained_for_pred/' + \
    model_type + '/model/scratch_model_classid.json'
train_log_path = './trained_for_pred/' + \
    model_type + '/model/log/model_train.csv'
train_checkpoint_path = './trained_for_pred/' + model_type + \
    '/model/log/Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.h5'
model_tensorboard_log = './training_log/tensorbord/'


# model training params
num_of_epoch = 50
num_of_train_samples = 3400
num_of_validation_samples = 600


# Cost function
model_loss_function = 'binary_crossentropy'


# define optimizers
model_optimizer_rmsprop = 'rmsprop'
model_optimizer_adam0 = 'adam'
model_optimizer_adam = Adam(lr=0.003, decay=0.00001)
model_optimizer_sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)


# model metrics to evaluate training
model_metrics = ["accuracy"]

# batch size
train_batch_size = 16
val_batch_size = 32

# for deleting a file


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        pass

# for saving model summary into a file


def save_summary(s):
    with open(model_summary_file, 'a') as f:
        f.write('\n' + s)
        f.close()
        pass


def main():
    # Init the class DataManager
    print("===================== load data =========================")
    dataManager = DataManager(img_height, img_width)
    # Get data
    train_data, validation_data = dataManager.get_train_data(
        train_data_dir, validation_data_dir, train_batch_size, val_batch_size)
    # Get class name:id
    label_map = (train_data.class_indices)
    # save model class id
    with open(saved_model_classid_path, 'w') as outfile:
        json.dump(label_map, outfile)
    # Init the class ScratchModel

    scratchModel = ScratchModel(image_shape, class_number)
    # Get model architecture

    print("===================== load model architecture =========================")
    model = scratchModel.get_model_architecture()
    # plot the model
    plot_model(model, to_file=model_png)  # not working with windows
    # serialize model to JSON
    model_json = model.to_json()
    with open(saved_model_arch_path, "w") as json_file:
        json_file.write(model_json)

    print("===================== compile model =========================")

    # Compile the model
    model = scratchModel.compile_model(
        model, model_loss_function, model_optimizer_rmsprop, model_metrics)



    '''
    # Delete the last summary file
    delete_file(model_summary_file)
    # Add the new model summary
    model.summary(print_fn=save_summary)

    # Prepare callbacks
    csv_log = callbacks.CSVLogger(train_log_path, separator=',', append=False)
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    checkpoint = callbacks.ModelCheckpoint(
        train_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(
        log_dir=model_tensorboard_log + "{}".format(time()))
    callbacks_list = [csv_log, tensorboard, checkpoint]

    print("===================== start training model =========================")
    # start training

    history = model.fit_generator(train_data,
                                  steps_per_epoch=num_of_train_samples // train_batch_size,
                                  epochs=num_of_epoch,
                                  validation_data=validation_data,
                                  validation_steps=num_of_validation_samples // val_batch_size,
                                  verbose=1,
                                  callbacks=callbacks_list)

    print(history)
    print("========================= training process completed! ===========================")
    '''


if __name__ == "__main__":
    main()

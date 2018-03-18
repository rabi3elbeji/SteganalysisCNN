##########################################################################
####################  This class is for evaluating a trained model  ######
##########################################################################

# import libraries
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, mean_absolute_error, mean_squared_error, classification_report, accuracy_score
from utils.data_manager import DataManager
from utils.plot_data import PlotData
from keras.models import model_from_json
from keras import backend as K
K.set_image_dim_ordering('tf')


# Cost function
model_loss_function = 'binary_crossentropy'


# define optimizers
model_optimizer_rmsprop = 'rmsprop'

# model metrics to evaluate training
model_metrics = ["accuracy"]

# batch size
batch_size = 16

# Load architecture and weights of the model


def load_model(model_arch_path, model_weights_path):
    json_file = open(model_arch_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights_path)
    return loaded_model


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        pass


# Model type
model_types = ['hugo_128_03', 'hugo_128_05', 'hugo_128_07', 'hugo_128_10', 'wow_128_03', 'wow_128_05', 'wow_128_07', 'wow_128_10', 'su_128_03', 'su_128_05', 'su_128_07', 'su_128_10']
legends = ['HUGO 0.3bpp', 'HUGO 0.5bpp', 'HUGO 0.7bpp', 'HUGO 1.0bpp', 'WOW 0.3bpp', 'WOW 0.5bpp', 'WOW 0.7bpp', 'WOW 1.0bpp', 'S-UNIWARD 0.3bpp', 'S-UNIWARD 0.5bpp', 'S-UNIWARD 0.7bpp', 'S-UNIWARD 1.0bpp']



linestyles = [':', '-.', '--', '-']
colors = ['g', 'r', 'b']

# Images width, height, channels
img_height = 128
img_width = 128
# number of test samples in the dataset
num_of_test_samples = 800

# for confusion matrix plotting
cm_plot_labels = ['cover', 'stego']
save_plt_roc = './model_roc_curves.png'


def main():
    y_trues = []
    y_preds = []

    for i in xrange(0, len(model_types)):
        model_type = model_types[i]

        # test dataset
        model_test_dataset = 'dataset_' + model_type
        # path to saved model files
        saved_model_weights_path = './trained_for_pred/' + \
            model_type + '/model/Best-weights.h5'
        saved_model_arch_path = './trained_for_pred/' + \
            model_type + '/model/scratch_model.json'
        test_data_dir = './datasets/' + model_test_dataset + '/test'

        # init DataManager class
        dataManager = DataManager(img_height, img_width)

        # load model
        print("===================== load model =========================")
        model = load_model(saved_model_arch_path, saved_model_weights_path)
        # get test data
        print("===================== load data =========================")
        test_data = dataManager.get_test_data(test_data_dir)
        # start the eval process
        print("===================== start eval =========================")
        y_true = test_data.classes
        # Confution Matrix and Classification Report
        Y_pred = model.predict_generator(
            test_data, num_of_test_samples // batch_size)
        y_pred = np.argmax(Y_pred, axis=1)

        y_trues.append(y_true)
        y_preds.append(y_pred)

    # init PlotData class
    plotData = PlotData()
    # Compute ROC curve and ROC area for each class
    plotData.plot_roc(y_trues, y_preds, colors, linestyles, legends, save_plt_roc)

if __name__ == "__main__":
    main()

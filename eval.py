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

# Model type
model_type = 'wow_128_04'

# Images width, height, channels
img_height = 128
img_width = 128
# number of test samples in the dataset
num_of_test_samples = 202

# for confusion matrix plotting
cm_plot_labels = ['cover', 'stego']

# test dataset
model_test_dataset = 'dataset_' + model_type

# path to saved model files
saved_model_weights_path = './trained_for_pred/' + \
    model_type + '/model/scratch_model.h5'
saved_model_arch_path = './trained_for_pred/' + \
    model_type + '/model/scratch_model.json'
test_data_dir = './datasets/' + model_test_dataset + '/test'

# paths to save outputs
save_plt_cm = './trained_for_pred/' + model_type + '/stats/scratch_model_cm.png'
save_plt_normalized_cm = './trained_for_pred/' + \
    model_type + '/stats/scratch_model_norm_cm.png'
save_plt_roc = './trained_for_pred/' + \
    model_type + '/stats/scratch_model_roc.png'
save_eval_report = './trained_for_pred/' + \
    model_type + '/stats/eval_report.txt'

train_log_data_path = './trained_for_pred/' + \
    model_type + '/model/log/model_train.csv'

save_plt_accuracy = './trained_for_pred/' + \
    model_type + '/stats/model_accuracy1.png'

save_plt_loss = './trained_for_pred/' + \
    model_type + '/stats/model_loss1.png'

save_plt_learning = './trained_for_pred/' + \
    model_type + '/stats/model_learning1.png'
# Cost function
model_loss_function = 'categorical_crossentropy'


# define optimizers
model_optimizer_rmsprop = 'rmsprop'

# model metrics to evaluate training
model_metrics = ["accuracy"]


# Load architecture and weights of the model
def load_model():
    json_file = open(saved_model_arch_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(saved_model_weights_path)
    return loaded_model


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        pass


def main():
    # init DataManager class
    dataManager = DataManager(img_height, img_width)
    # init PlotData class
    plotData = PlotData()
    # load model
    print("===================== load model =========================")
    model = load_model()
    # get test data
    print("===================== load data =========================")
    test_data = dataManager.get_test_data(test_data_dir)
    # start the eval process
    print("===================== start eval =========================")
    y_true = test_data.classes
    # Confution Matrix and Classification Report
    Y_pred = model.predict_generator(test_data)
    y_pred = np.argmax(Y_pred, axis=1)
    # plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plotData.plot_confusion_matrix(
        cm, cm_plot_labels, save_plt_cm, title='Confusion Matrix')
    plotData.plot_confusion_matrix(
        cm, cm_plot_labels, save_plt_normalized_cm, normalize=True, title='Normalized Confusion Matrix')
    # Compute ROC curve and ROC area for each class
    roc_auc = plotData.plot_roc(y_true, y_pred, save_plt_roc)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print('mean absolute error: ' + str(mae))
    print('mean squared error: ' + str(mse))
    print('Area Under the Curve (AUC): ' + str(roc_auc))
    c_report = classification_report(
        y_true, y_pred, target_names=cm_plot_labels)
    print(c_report)
    delete_file(save_eval_report)
    with open(save_eval_report, 'a') as f:
        f.write('\n\n')
        f.write('******************************************************\n')
        f.write('**************   Evalaluation Report   ***************\n')
        f.write('******************************************************\n')
        f.write('\n\n')
        f.write('- Accuracy Score: ' + str(accuracy))
        f.write('\n\n')

        f.write('- Mean Absolute Error (MAE): ' + str(mae))
        f.write('\n\n')

        f.write('- Mean Squared Error (MSE): ' + str(mse))
        f.write('\n\n')

        f.write('- Area Under the Curve (AUC): ' + str(roc_auc))
        f.write('\n\n')

        f.write('- Confusion Matrix:\n')
        f.write(str(cm))
        f.write('\n\n')

        f.write('- Normalized Confusion Matrix:\n')
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        f.write(str(cm))
        f.write('\n\n')

        f.write('- Classification report:\n')
        f.write(str(c_report))

        f.close()

    train_validation = ['train', 'validation']
    data = pd.read_csv(train_log_data_path)
    acc = data['acc'].values
    val_acc = data['val_acc'].values
    loss = data['loss'].values
    val_loss = data['val_loss'].values

     
    # plot metrics to the stats dir
    plotData.plot_2d(acc, val_acc, 'epoch', 'accuracy',
                     'Model Accuracy', train_validation, save_plt_accuracy)
    plotData.plot_2d(loss, val_loss, 'epoch', 'loss',
                     'Model Loss', train_validation, save_plt_loss)
    plotData.plot_model_bis(data, save_plt_learning)
    
    '''
    # evalute model
    # compile the model
    print("==================== compile model ========================")
    model.compile(loss=model_loss_function, optimizer=model_optimizer_rmsprop, metrics=model_metrics)
    
   
    score = model.evaluate_generator(test_data, num_of_test_samples)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])
    '''


if __name__ == "__main__":
    main()

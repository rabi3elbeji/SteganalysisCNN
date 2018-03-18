###############################################################################################
####################  This class is for plotting specefic data and not part of the code  ######
###############################################################################################

# import libraries
import numpy as np
import pandas as pd
import os
from utils.plot_data import PlotData


# Model type
model_type = 'wow_128_05'
# path to save plots
save_plt_learning = './model_without_transfer_learning.eps'
# path to data
train_log_data_path = './trained_for_pred/' + \
    model_type + '/model/log/model_train.csv'

train_log_data_path = './trained_for_pred_bis/model_train.csv'

def main():
    # init PlotData class
    plotData = PlotData()
    # read data
    data = pd.read_csv(train_log_data_path)
    linestyles = [':', '-', '-.', ':']
    colors = ['b', 'r', 'g']
    markers = ['^', 'D']
    legend = ['Trn 0.7bpp', 'Val 0.7bpp', 'Trn 0.5bpp', 'Val 0.5bpp', 'Trn 0.3bpp', 'Val 0.3bpp']
    plotData.plot_custom_data(data, colors, linestyles, markers, legend, save_plt_learning)

    print(data)



if __name__ == "__main__":
    main()
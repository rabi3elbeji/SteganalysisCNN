##########################################################################
####################  This class is for plotting specefic data and not par
##########################################################################

# import libraries
import numpy as np
import pandas as pd
import os
from utils.plot_data import PlotData


save_plt_learning = './detection_error_compare.eps'


def main():
    plotData = PlotData()
    linestyles = ['-', '-', '-.', ':']
    colors = ['b', 'r', 'g']
    markers = ['^', 'D', 'o']
    legend = ['S-UNIWARD', 'WOW', 'HUGO']
    palyloads = [1, 0.7, 0.5, 0.3]
    errors = [[0.04, 0.07, 0.12, 0.27], [0.04, 0.08, 0.17, 0.33], [0.04, 0.09, 0.16, 0.31]]
    plotData.plot_detection_error(
        palyloads, errors, colors, linestyles, markers, legend, save_plt_learning)


if __name__ == "__main__":
    main()

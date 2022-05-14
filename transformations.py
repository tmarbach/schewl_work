import numpy as np
from dataset_defintions import *

def transform_xy(windows, class_dict):
    """
    desc:
        Converts list of single class dataframes to two arrays
        Xdata (all raw/transformed datapoints) & ydata (class label).
        transformations included per axis :
            mean, std, min, max, kurtosis, skew, corr (xy, yz, xz)
    params:
        windows -- list of dataframes of all one class
    return:
        Xdata -- arrays of xyz data of each window stacked together
        ydata -- integer class labels for each window
    """
    positions = [ACCELERATION_X, ACCELERATION_Y, ACCELERATION_Z]

    Xdata, ydata = [], []
    for window in windows:
        alldata = np.empty((0,3), int)
        alldata = np.append(alldata, np.float32([window[positions].mean(axis = 0)]), 0)
        alldata = np.append(alldata, np.float32([window[positions].std(axis = 0)]),  0)

        Xdata.append(alldata)
        behavior = window[BEHAVIOR].iloc[0]

        # Bucketing the behavior to be categorized as strike
        if behavior in STRIKES:
            # t designates a general strike behavior
            behavior = 't'

        ydata.append(class_dict[behavior])

    return np.stack(Xdata), np.asarray(ydata)
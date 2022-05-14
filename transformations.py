import numpy as np
from dataset_defintions import *

def transform_accel_xyz(windows, class_dict):
    """
    desc:
        Converts list of single class dataframes to two arrays
        Xdata (all raw/transformed datapoints) & ydata (class label).
        transformations included per axis :
            mean, std, min, max, kurtosis, skew, corr (xy, yz, xz)
    params:
        windows -- list of dataframes of all one class
    return:
        Xdata -- arrays of x,y,z data of each window stacked together
        ydata -- integer class labels for each window
    """
    positions = [ACCELERATION_X, ACCELERATION_Y, ACCELERATION_Z]

    Xdata, ydata = [], []
    for window in windows:
        data = np.empty((0,3), int)
        # Take the mean and standard deviation of each X, Y, Z
        data = np.append(data, np.float32([window[positions].mean(axis = 0)]), 0)
        data = np.append(data, np.float32([window[positions].std(axis = 0)]),  0)

        Xdata.append(data)
        behavior = window[BEHAVIOR].iloc[0]

        # Bucketing the behavior to be categorized as strike
        if behavior in STRIKES:
            # t designates a general strike behavior
            behavior = 't'

        ydata.append(class_dict[behavior])

    return np.stack(Xdata), np.asarray(ydata)
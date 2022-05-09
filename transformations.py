import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def transform_xy(windows, classdict):
    """
    Purpose:
        Converts list of single class dataframes to two arrays
        Xdata (all raw/transformed datapoints) & ydata (class label).
        transformations included per axis :
            mean, std, min, max, kurtosis, skew, corr (xy, yz, xz)
    Input:
        windows -- list of dataframes of all one class
    Output:
        Xdata -- arrays of xyz data of each window stacked together
        ydata -- integer class labels for each window
    """
    positions = ['accX', 'accY', 'accZ']
    strikes = ['h', 'm']

    Xdata, ydata = [], []
    for window in windows:
        alldata = np.empty((0,3), int)
        alldata = np.append(alldata, np.float32([window[positions].mean(axis = 0)]), 0)
        alldata = np.append(alldata, np.float32([window[positions].std(axis = 0)]), 0)

        Xdata.append(alldata)
        behavior = window['behavior'].iloc[0]
        if behavior in strikes:
                behavior  = 't'
        ydata.append(classdict[behavior])
    return np.stack(Xdata), np.asarray(ydata)
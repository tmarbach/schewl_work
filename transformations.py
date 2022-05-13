import numpy as np
from dataset_defintions import *
from sliding_window import BEHAVIOR

def bucket_strikes(df):
    # Bucketing the behavior to be categorized as strike
    df[BEHAVIOR] = df[BEHAVIOR].replace(STRIKES, 't')
    return df

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
    positions = ['ACCX', 'ACCY', 'ACCZ']

    Xdata, ydata = [], []
    for window in windows:
        alldata = np.empty((0,3), int)
        alldata = np.append(alldata, np.float32([window[positions].mean(axis = 0)]), 0)
        alldata = np.append(alldata, np.float32([window[positions].std(axis = 0)]), 0)

        Xdata.append(alldata)
        behavior = window[BEHAVIOR].iloc[0]

        # Bucketing the behavior to be categorized as strike
        if behavior in STRIKES:
            behavior = 't'

        ydata.append(classdict[behavior])

    # Double check what asArray does
    return np.stack(Xdata), np.asarray(ydata)
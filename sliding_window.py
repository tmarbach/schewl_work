import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# class SamplePreparation:
#     """
#     Given a Sliding Window strategy and a Sampling Strategy prepare data
#     """

BEHAVIOR = 'BEHAVIOR'

def prepared_samples(Xdata, ydata, sample_flag=None):
    x_data_reduced = reduce_sample_dimension(Xdata)
    x_train, x_test, y_train, y_test = stratified_shuffle_split(
        x_data_reduced, ydata)

    # reduce dimension sampler only does the sampling technique on the training data.
    x_train_resampled, y_train_resampled = sample_data(x_train, y_train, sample_flag)

    return x_train_resampled, x_test, y_train_resampled, y_test


def check_data_size(window_size, data_shape):
    if window_size > data_shape:
        raise ValueError('Window larger than data given')

def singleclass_leaping_window_exclusive(df, window_size, coi=False):
    """
    desc: 

    params: 
        df -- dataframe of cleaned input data, likely from a csv
        window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)

    return:
        windows -- list of lists of accel data (EX:[x,y,z,...,x,y,z,class_label])
        allclasses -- list of the behavior classes that are present in the windows
    """
    classes = []
    windows = []

    data_shape = df.shape[0]
    check_data_size(window_size, data_shape)

    number_of_rows_minus_window = data_shape - window_size + 1
    for i in range(0, number_of_rows_minus_window, window_size):
        window = df[i: i+window_size]

        if len(window[BEHAVIOR].unique()) != 1:
            continue
        if len(set(np.ediff1d(window.index))) != 1:
            continue

        if coi:
            coi_classes = list(coi)
            current_behavior = window[BEHAVIOR].iloc[0]
            if current_behavior not in list(coi_classes):
                continue

        windows.append(window)
        classes.append(window[BEHAVIOR].iloc[0])

    allclasses = set(classes)
    if coi:
        diff = list(set(coi) - allclasses)
        if len(diff) > 0:
            missingclasses = ','.join(str(c) for c in diff)
            #print("Classes " + missingclasses + " not found in any window.")

    #print("Windows pulled")
    return windows, list(allclasses)

def reduce_sample_dimension(X_data):
    """
    desc
        reduces the dimensions of the raw data in preperation for training and testing
    param
        X_data - raw sample data
    return
        X_data_2d with sample dimensions reduced from 3D to 2D
    """
    samples, x, y = X_data.shape
    # Reduce the dimensions from 3D to 2D
    X_data_2d = X_data.reshape((samples, x * y))
    return X_data_2d

def stratified_shuffle_split(Xdata, ydata):
    """
    desc
        the point of doing this to make sure that strike data makes it guarantees it makes it to test stage
    param

    return
    """
    stshsp = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_index, test_index = next(stshsp.split(Xdata, ydata))
    x_train, x_test = Xdata[train_index], Xdata[test_index]
    y_train, y_test = ydata[train_index], ydata[test_index]

    return x_train, x_test, y_train, y_test,

def sample_data(x_data, y_data, sample_flag):
    """
        Reduces the dimensions of inout data, splits into train/test set,
        then randomly oversamples the minority classes to match the majority class

    """
    if sample_flag == 'o':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(x_data, y_data)
    elif sample_flag == 's':
        X_resampled, y_resampled = SMOTE(
            k_neighbors=2).fit_resample(x_data, y_data)
    elif sample_flag == 'a':
        X_resampled, y_resampled = ADASYN(
            n_neighbors=2).fit_resample(x_data, y_data)
    else:
        # If no sample flag has been specified or the flag is not recognized,
        # do not apply a sample technique
        X_resampled, y_resampled = x_data, y_data

    return X_resampled, y_resampled

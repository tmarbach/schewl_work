"""
File Overview
--------------
Function for preparing a cleaned acceleration dataframe for training and testing with a Model
"""

from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

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

def stratified_shuffle_split(X_data, y_data):
    """
    desc
        Create training and testing sets using stratified shuffle technique.
        Stratified shuffle ensures strike data makes it to test stage
    param
        X_data - Sample features
        y_data - Sample labels
    return
        train test split of the data and labels
    """
    split_strategy = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_index, test_index = next(split_strategy.split(X_data, y_data))
    x_train, x_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    return x_train, x_test, y_train, y_test,

def apply_sampling(X_data, y_data, sample_flag):
    """
    desc
        then randomly oversamples the minority classes to match the majority class
    params
        X_data      - sample data
        y_data      - ground truth of the samples
        sample_flag - designation of what sampling strategy should be used
    return
        X_resampled - sample data with sampling technique applied
        y_resampled - resized array of the ground truth labels to match sampled data
    """
    if sample_flag == 'o':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X_data, y_data)
    elif sample_flag == 's':
        X_resampled, y_resampled = SMOTE(k_neighbors=2).fit_resample(X_data, y_data)
    elif sample_flag == 'a':
        X_resampled, y_resampled = ADASYN(n_neighbors=2).fit_resample(X_data, y_data)
    else:
        # If no sample flag has been specified or the flag is not recognized,
        # do not apply a sample technique
        X_resampled, y_resampled = X_data, y_data

    return X_resampled, y_resampled

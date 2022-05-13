"""
File Overview
--------------
Entry point for running a selected model and sampling technique against the acceleration dataset
"""

import argparse

# Data Read and Initial Clean-up
from clean_acceleration_data import clean_dataset 

# Constants used to refer to the dataset post clean-up
from dataset_defintions import *

# Data Tranformations and Model Preparation
from prepare_acceleration_data import reduce_sample_dimension, stratified_shuffle_split, apply_sampling
from window_maker import singleclass_leaping_window
from transformations import transform_xy

# Models
from model import naive_bayes, svm, random_forest

# Location of dataset
PATH = "./dataset_subset/"

def retrieve_arguments():
    """
    
    """
    parser = argparse.ArgumentParser(
            prog='model_select', 
            description="Select a ML model to apply to acceleration data",
            epilog="") 
    parser.add_argument(
            "-m",
            "--model",
            help = "Choose a ML model rf, svm, naive bayes",
            default="rf", 
            type=str)
    parser.add_argument(
            "-o",
            "--oversample",
            help = "Flag to oversample the minority classes: o -- oversample, s -- SMOTE, or a -- ADASYN ",
            default='ns', 
            type=str 
            )
    return parser.parse_args()


def run_model(model_selection, X_train, X_test, y_train, y_test):
    """
    
    """
    if model_selection == 'svm':
        report, parameter_list, classes = svm(X_train, X_test, y_train, y_test)
    elif model_selection == 'nb':
        report, parameter_list, classes = naive_bayes(X_train, X_test, y_train, y_test)
    else: # Default to random forest model
        report, parameter_list, classes = random_forest(X_train, X_test, y_train, y_test, CLASSES_OF_INTEREST_LIST)

    return report, parameter_list, classes


def prepare_train_test_pipeline(windowed_X_data, y_data, sample_flag=None):
    """
    desc
        Pipeline created to prepare windowed data, apply sampling techniques and
        split into test and train. 
    params
        windowed_X_data - X data that has had a windowing technique applied
        y_data          - ground truth
        sample_flag     - sampling technique to be applied

    """
    x_data_reduced = reduce_sample_dimension(y_data)
    x_train, x_test, y_train, y_test = stratified_shuffle_split(
        x_data_reduced, y_data)

    # reduce dimension sampler only does the sampling technique on the training data.
    x_train_resampled, y_train_resampled = apply_sampling(x_train, y_train, sample_flag)

    return x_train_resampled, x_test, y_train_resampled, y_test

def main(args):
    df = clean_dataset(PATH)

    windows, classes_found = singleclass_leaping_window(df, WINDOW_SIZE, CLASSES_OF_INTEREST)
    Xdata, ydata = transform_xy(windows, CLASS_INDICES)
    X_train, X_test, y_train, y_test = prepare_train_test_pipeline(Xdata, ydata, args.oversample)

    # Run Model based on argument selection
    report, y_pred, classes = run_model(args.model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    args = retrieve_arguments()
    main(args)


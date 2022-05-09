#!/usr/bin/env python3

import pandas as pd
import argparse

# Data Read and Initial Clean-up
from accelml_prep_csv import prepare_dataset 

# Data Tranformations and Model Preparation
from sliding_window import singleclass_leaping_window_exclusive, reduce_dim_sampler
from transformations import transform_xy
from rater_converter import rater_construct_train_test, accel_singlelabel_xy, accel_oversampler

# Models
from model import naive_bayes, svm, random_forest

def retrieve_arguments():
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
            default=False, 
            type=str 
            )
    return parser.parse_args()


def run_model(model_selection, X_train, X_test, y_train, y_test, n_classes):
    """
    
    """
    if model_selection == 'svm':
        report, parameter_list = svm(X_train, X_test, y_train, y_test)
    elif model_selection == 'nb':
        report, parameter_list = naive_bayes(X_train, X_test, y_train, y_test)
    else: # Default to random forest model
        report, parameter_list = forester(X_train, X_test, y_train, y_test, n_classes)

    return report, parameter_list

WINDOW_SIZE = 25
CLASSES_OF_INTEREST = "hlmstw"
CLASS_INDICES = {
    0: 'h',
    1: 'l',
    2: 'm',
    3: 's',
    4: 't',
    5: 'w',
}

PATH = "./dataset/"

def main(args):
    df = prepare_dataset(PATH)

    windows, actual_classes = singleclass_leaping_window_exclusive(df, WINDOW_SIZE, CLASSES_OF_INTEREST)
    Xdata, ydata = transform_xy(windows, CLASS_INDICES)
    X_train, X_test, y_train, y_test = reduce_dim_sampler(Xdata, ydata, args.oversample)

    # Run Model based on argument selection
    report, parameter_list = run_model(X_train, X_test, y_train, y_test, len(CLASSES_OF_INTEREST))
    reportdf = pd.DataFrame(report).transpose()

    # EXTRAS for Reporting
    #n_samples, n_features, n_classes = Xdata.shape[0], Xdata.shape[1]*Xdata.shape[2], len(actual_classes) # keep this line to compare to each time we run how the oversampling techniques get done
    # key = construct_key(args.model, WINDOW_SIZE)
    # output_data(reportdf,args.model, key, args.data_output_file)
    # recall_matrixdf = pd.DataFrame(recall_matrix, index = actual_classes,columns = actual_classes)
    # recall_matrixdf.to_csv(args.label_output_file)
    # precision_matrixdf = pd.DataFrame(precision_matrix, index = actual_classes,columns = actual_classes)
    # precision_matrixdf.to_csv(args.param_output_file)

if __name__ == "__main__":
    args = retrieve_arguments()
    main(args)


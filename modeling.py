import pandas as pd
import argparse

# Data Read and Initial Clean-up
from clean_acceleration_data import clean_dataset 

# Data Tranformations and Model Preparation
from sliding_window import singleclass_leaping_window_exclusive, prepare_train_test_data
from transformations import transform_xy
from dataset_defintions import *

# Models
from model import naive_bayes, svm, random_forest

# Metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

PATH = "./dataset_subset/"

def main(args):
    # Int
    df = clean_dataset(PATH)

    windows, classes_found = singleclass_leaping_window_exclusive(df, WINDOW_SIZE, CLASSES_OF_INTEREST)
    Xdata, ydata = transform_xy(windows, CLASS_INDICES)
    X_train, X_test, y_train, y_test = prepare_train_test_data(Xdata, ydata, args.oversample)

    # Run Model based on argument selection
    report, y_pred, classes = run_model(args.model, X_train, X_test, y_train, y_test)


    # 
    model_name = MODEL_NAMES[args.model]
    sampling_technique_name = SAMPLING_TECHNIQUE_NAMES[args.oversample]
    configuration = str(model_name + '_' + sampling_technique_name)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES_OF_INTEREST_LIST)
    display.plot()
    display.ax_.set_title(configuration)
    plt.savefig(str("./figures/" + configuration + '.png'))

    report = pd.DataFrame(report).transpose()

    result_path = str('./results/' + configuration + '.xlsx') 
    with pd.ExcelWriter(result_path, engine='xlsxwriter') as writer:
        report.to_excel(writer, index=False)


if __name__ == "__main__":
    args = retrieve_arguments()
    main(args)


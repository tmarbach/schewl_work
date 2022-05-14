"""
File Overview
--------------
Contains all modeling stratgies to be used with Acceleration Data
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def random_forest(X_train, X_test, y_train, y_test, clases_names):
    """
    desc
        Run the Random Forest algorithm against the given test and train data.
    params
        X_train     - samples to train on
        X_test      - samples to test on
        y_train     - ground truth of training data
        y_test      - ground truth of test data
        parameters  - parameters for the SVM model, default it given
    return
        report  -   includes accuracy, percision per-class, 
                    recall per-class, macro average and weigted average
    """
    rf_clf=RandomForestClassifier(
        n_estimators=1000,
        max_leaf_nodes=len(clases_names),
        n_jobs=-1)

    rf_clf.fit(X_train, y_train)
    y_pred=rf_clf.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=clases_names,
        output_dict=True)

    return report, y_pred, rf_clf.classes_


def naive_bayes(X_train, X_test, y_train, y_test):
    """
    desc
        Run the Naive Bayes algorithm against the given test and train data.
    params
        X_train     - samples to train on
        X_test      - samples to test on
        y_train     - ground truth of training data
        y_test      - ground truth of test data
        parameters  - parameters for the SVM model, default it given
    return
        report  -   includes accuracy, percision per-class, 
                    recall per-class, macro average and weigted average
    """
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    nb_clf = ComplementNB()
    nb_clf.fit(X_train, y_train)
    y_pred = nb_clf.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True)

    return report, y_pred, nb_clf.classes_


def svm(X_train, X_test, y_train, y_test, parameters={"degrees": 3, "C": 5, "kernel":"poly"}):
    """
    desc
        Run the Support Vector Machine algorithm against the given test and train data.
    params
        X_train     - samples to train on
        X_test      - samples to test on
        y_train     - ground truth of training data
        y_test      - ground truth of test data
        parameters  - parameters for the SVM model, default it given
    return
        report  -   includes accuracy, percision per-class, 
                    recall per-class, macro average and weigted average
    """
    svm_clf = Pipeline([
            ("scalar", StandardScaler()),
            ("linear_svc", 
            SVC(kernel = parameters["kernel"],
                degree = parameters["degrees"],
                C = parameters["C"])),
    ])
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    report = classification_report(
        y_test,
        y_pred, 
        output_dict=True)

    return report, y_pred, svm_clf.classes_
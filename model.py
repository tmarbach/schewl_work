
"""
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
        target_names = clases_names,
        output_dict=True)

    parameters = rf_clf.get_params()

    return report, parameters


def naive_bayes(X_train, X_test, y_train, y_test):
    """
    desc

    params
    

    return 
        return report,
        parameters (dict with parameter names mapped to values)
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

    parameters = nb_clf.get_params()
    return report, parameters


def svm(X_train, X_test, y_train, y_test, parameters={"degrees": 3, "C": 5, "kernel":"poly"}):
    """
    
    """
    svm_clf = Pipeline([
            ("scalar", StandardScaler()),
            ("linear_svc", 
            SVC(kernel = parameters["kernel"],
                degree = parameters["degrees"],
                C = parameters["C"])),
    ])
    svm_clf.fit(X_train, y_train)
    ypred = svm_clf.predict(X_test)
    report = classification_report(
        y_test,
        ypred, 
        output_dict=True)
    parameters = svm_clf.get_params()
    
    # return report, parameters (dict with parameter names mapped to values)
    return report, parameters
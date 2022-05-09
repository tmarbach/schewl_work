from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def svm(X_train, X_test, y_train, y_test):
    parameters = {"degrees":3, "C":5, "kernel":"poly"}
    svm_clf = Pipeline([
            ("scalar", StandardScaler()),
            ("linear_svc", 
            SVC(kernel=parameters["kernel"],
                degree=parameters["degrees"],
                C=parameters["C"])),
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
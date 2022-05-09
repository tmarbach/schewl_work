from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler


def naive_bayes(X_train, X_test, y_train, y_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = ComplementNB()
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)
    report = classification_report(
        y_test,
        ypred, 
        output_dict=True
        )
    parameters = clf.get_params()
    # return report, parameters (dict with parameter names mapped to values)
    return report, parameters

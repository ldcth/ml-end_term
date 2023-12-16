
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

from sklearn.svm import SVC
def svm(x_train, x_test, y_train, y_test):
    # classifier =  SVC(kernel="rbf", C=10000)
    classifier = SVC()

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm

def svm_rbf(x_train, x_test, y_train, y_test):
    classifier =  SVC(kernel="rbf")
    # classifier = SVC()

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm

def svm_rbf_c(x_train, x_test, y_train, y_test):
    classifier =  SVC(kernel="rbf", C=1000)

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    joblib.dump(classifier, './checkpoint/svm_rbf_c_classifier.joblib')

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")

    # Confusion Matrixs
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm
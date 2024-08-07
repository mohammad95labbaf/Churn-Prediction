# classification.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def define_classifiers(random_state):
    classifiers = {
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=random_state),
        "Voting": VotingClassifier(estimators=[
            ('dt', DecisionTreeClassifier(random_state=random_state)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('svm', SVC(probability=True, random_state=random_state))
        ], voting='soft'),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Bagging": BaggingClassifier(n_estimators=100, random_state=random_state),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=random_state, algorithm='SAMME'),
    }
    return classifiers

def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    return accuracy, cm, cr

def print_classification_results(classifier_name, accuracy, cm, cr):
    print(f"\n{classifier_name} Classifier:")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
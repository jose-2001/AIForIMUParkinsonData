from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def build_models_modules(x, y):
    svm = __train_svm(x, y)
    knn = __train_knn(x, y)
    gb = __train_gboost(x, y)
    return svm, knn, gb


def build_models_general_classifier(x, y):
    dt = __train_dt(x, y)
    rf = __train_rf(x, y)
    return dt, rf


def __train_svm(x, y):
    svm = SVC()
    parameters = {
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'C': [1, 10, 100, 1000]
    }
    clf = GridSearchCV(svm, parameters)
    clf.fit(x, y)
    return clf


def __train_knn(x, y):
    knn = KNeighborsClassifier()
    parameters = {
        'n_neighbors': [3, 5, 7, 11],
        'weights': ['uniform', 'distance']
    }
    clf = GridSearchCV(knn, parameters)
    clf.fit(x, y)
    return clf


def __train_gboost(x, y):
    gboost = GradientBoostingClassifier()
    parameters = {
        'learning_rate': [0.1, 0.01, 0.001, 0.0001],
        'n_estimators': [24, 48, 64, 128],
        'n_iter_no_change': [20]
    }
    clf = GridSearchCV(gboost, parameters)
    clf.fit(x, y)
    return clf


def __train_dt(x, y):
    dt = DecisionTreeClassifier()
    dt.fit(x, y)
    return dt


def __train_rf(x, y):
    rf = RandomForestClassifier()
    parameters = {
        'n_estimators': [100, 150, 200, 300]
    }
    clf = GridSearchCV(rf, parameters)
    clf.fit(x, y)
    return clf

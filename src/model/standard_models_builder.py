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
    print('Started Grid Search - Model: SVM')
    clf = GridSearchCV(svm, parameters, n_jobs=-1, verbose=4)
    clf.fit(x, y)
    print("Best parameters SVM: {}".format(clf.best_params_))
    print('Finished Grid Search - Model: SVM')
    return clf


def __train_knn(x, y):
    knn = KNeighborsClassifier()
    parameters = {
        'n_neighbors': [3, 5, 7, 11],
        'weights': ['uniform', 'distance']
    }
    print('Started Grid Search - Model: KNN')
    clf = GridSearchCV(knn, parameters, n_jobs=-1, verbose=4)
    clf.fit(x, y)
    print("Best parameters KNN: {}".format(clf.best_params_))
    print('Finished Grid Search - Model: KNN')
    return clf


def __train_gboost(x, y):
    gboost = GradientBoostingClassifier()
    parameters = {
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [24, 48, 64],
        'n_iter_no_change': [20]
    }
    print('Started Grid Search - Model: Gradient Boosting Classifier')
    clf = GridSearchCV(gboost, parameters, n_jobs=-1, verbose=4)
    clf.fit(x, y)
    print("Best parameters Gradient Boosting Classifier: {}".format(clf.best_params_))
    print('Finished Grid Search - Model: Gradient Boosting Classifier')
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
    clf = GridSearchCV(rf, parameters, n_jobs=-1, verbose=4)
    clf.fit(x, y)
    return clf

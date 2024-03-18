from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def create_svc_tremor_model(scaler: bool = False, gamma: float | str = 1, kernel: str = 'rbf'):
    model = SVC(kernel=kernel, gamma=gamma)
    pipeline = make_pipeline(model)
    if scaler:
        pipeline = make_pipeline(StandardScaler(), model)
    return pipeline


def create_knn_tremor_model(scaler: bool = False, k: int = 3):
    model = KNeighborsClassifier(n_neighbors=k)
    pipeline = make_pipeline(model)
    if scaler:
        pipeline = make_pipeline(StandardScaler(), model)
    return pipeline

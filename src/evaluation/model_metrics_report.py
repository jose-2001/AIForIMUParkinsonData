import numpy as np
import seaborn as sns
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc,
                             confusion_matrix, cohen_kappa_score)
import matplotlib.pyplot as plt


def print_model_metrics(model, x_test: np.ndarray, y_test: np.ndarray, contains_conv: bool = False):

    y_pred = model.predict(x_test)

    if contains_conv:
        arr = []
        for sample in y_pred:
            arr.append(sample[1])
        y_pred = np.array(arr)

    y_pred_binary = np.argmax(y_pred, axis=1)

    _get_accuracy(y_test, y_pred_binary)
    print(cohen_kappa_score(y_test, y_pred_binary))
    print(classification_report(y_test, y_pred_binary))
    _get_auc_roc(y_test, y_pred)
    _get_auc_pr(y_test, y_pred)
    plot_roc_curve(y_test, y_pred)
    plot_pr_curve(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred_binary)


def _get_accuracy(y_test: np.ndarray, y_pred: np.ndarray):
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)


def _get_auc_roc(y_test: np.ndarray, y_pred: np.ndarray):
    auc_roc = roc_auc_score(y_test, y_pred[:, 0])
    print("AUC-ROC:", auc_roc)


def _get_auc_pr(y_test: np.ndarray, y_pred: np.ndarray):
    fpr, tpr, _ = roc_curve(y_test, y_pred[:, 0])
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_pred[:, 0])
    auc_pr = auc(recall, precision)
    print("AUC-PR:", auc_pr)

    return fpr, tpr, roc_auc


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def plot_pr_curve(y_test: np.ndarray, y_pred: np.ndarray):
    precision, recall, _ = precision_recall_curve(y_test, y_pred[:, 0])
    auc_pr = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % auc_pr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()


def plot_roc_curve(y_test: np.ndarray, y_pred: np.ndarray):
    fpr, tpr, _ = roc_curve(y_test, y_pred[:, 0])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

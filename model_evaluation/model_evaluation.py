import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

class ModelEvaluator:
    def __init__(self, model: tf.keras.Model):
        self.model = model

    def evaluate(self, x_test, y_test, average="weighted"):
        """
        x_test: features (numpy array)
        y_test: one-hot labels (numpy array)
        average: "macro" | "micro" | "weighted"
        """
        # 1) Keras evaluate (loss/accuracy)
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)

        # 2) Predictions -> class labels
        y_prob = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # 3) Metrics
        metrics = {
            "loss": float(loss),
            "accuracy": float(acc),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, zero_division=0)
        }

        return metrics

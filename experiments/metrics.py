from statistics import mean
import json

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class Metrics:
    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f_score = []

    def add_new(self, preds: np.ndarray, trues: np.ndarray, losses):
        self.loss.append(mean(losses))
        precision, recall, f_scr, _ = precision_recall_fscore_support(
            trues,
            preds,
            average='weighted',
            zero_division=1
        )
        self.precision.append(precision)
        self.recall.append(recall)
        self.f_score.append(f_scr)
        self.accuracy.append(accuracy_score(trues, preds))

    def as_dict(self, current_state=False):
        return {
            "loss": self.loss if not current_state else self.loss[-1],
            "acc": self.accuracy if not current_state else self.loss[-1],
            "precision": self.precision if not current_state else self.loss[-1],
            "recall": self.recall if not current_state else self.loss[-1],
            "f": self.f_score if not current_state else self.loss[-1]
        }

    def __str__(self):
        metrics_dict = self.as_dict(current_state=True)
        return json.dumps(metrics_dict, indent=2)

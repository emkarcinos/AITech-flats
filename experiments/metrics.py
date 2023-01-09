from statistics import mean

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class Metrics:
    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f_score = []

    def add_new(self, preds: np.ndarray, trues: np.ndarray, losses: list[float]):
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

    def as_dict(self):
        return {
            "loss": self.loss,
            "acc": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f": self.f_score
        }

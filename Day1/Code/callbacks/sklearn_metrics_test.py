import pytorch_lightning as pl
from sklearn.metrics import classification_report, confusion_matrix


class TestMetricsSklearn(pl.Callback):
    def __init__(self):
        self.test_predictions = []
        self.test_labels = []

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.test_predictions.extend(outputs["pred"])
        self.test_labels.extend(outputs["label"])

    def on_test_epoch_end(self, trainer, pl_module):
        print("Classification report:")
        print(classification_report(
            self.test_predictions,
            self.test_labels
        ))

        print("Confusion matrix:")
        print(confusion_matrix(
            self.test_predictions,
            self.test_labels
        ))
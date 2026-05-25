import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class MetricsTop:
    def __init__(self, train_mode):
        if train_mode != "regression":
            raise ValueError("IVID currently supports regression metrics only.")

    def _multiclass_acc(self, y_pred, y_true):
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def _eval_regression(self, y_pred, y_true):
        preds = y_pred.view(-1).cpu().detach().numpy()
        truth = y_true.view(-1).cpu().detach().numpy()

        preds_a7 = np.clip(preds, a_min=-3.0, a_max=3.0)
        truth_a7 = np.clip(truth, a_min=-3.0, a_max=3.0)
        preds_a5 = np.clip(preds, a_min=-2.0, a_max=2.0)
        truth_a5 = np.clip(truth, a_min=-2.0, a_max=2.0)

        mae = np.mean(np.absolute(preds - truth)).astype(np.float64)
        corr = np.corrcoef(preds, truth)[0][1]
        mult_a7 = self._multiclass_acc(preds_a7, truth_a7)
        mult_a5 = self._multiclass_acc(preds_a5, truth_a5)

        valid_indices = np.array([i for i, value in enumerate(truth) if value != 0])
        valid_truth = truth[valid_indices] > 0
        valid_preds = preds[valid_indices] > 0

        acc = accuracy_score(valid_preds, valid_truth)
        f1 = f1_score(valid_truth, valid_preds, average="weighted")

        return {
            "Acc": round(acc, 4),
            "F1": round(f1, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),
        }

    def get_metrics(self, dataset_name):
        dataset_name = dataset_name.upper()
        if dataset_name not in {"MOSI", "MOSEI"}:
            raise ValueError("dataset_name must be MOSI or MOSEI")
        return self._eval_regression

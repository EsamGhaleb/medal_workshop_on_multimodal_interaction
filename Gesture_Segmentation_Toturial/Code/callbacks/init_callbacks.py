import os

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def setup_early_stopping_callback(metric, min_delta=0.00, patience=20, mode="min"):
    return EarlyStopping(monitor=metric, min_delta=min_delta, patience=patience, verbose=False, mode=mode)


def setup_model_checkpoint_callback(model_weights_path, metric, dataset, model, experiment_id, save_top_k=1):
    return ModelCheckpoint(
        monitor=metric,
        dirpath=os.path.join(model_weights_path, f"{dataset}-{model}-{experiment_id}"),
        filename="{epoch}",
        save_top_k=save_top_k,
        mode="max" if "loss" not in metric else "min",
        save_last=True
    )

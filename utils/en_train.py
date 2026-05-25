import copy
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.data_loader import data_loader
from utils.ivid_model import IVIDModel
from utils.metricsTop import MetricsTop


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dict_to_str(metrics):
    return " ".join(f"{key}: {value:.4f}" for key, value in metrics.items())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_logger(log_path, seed):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ivid")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt=f"%(asctime)s | seed={seed} | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


class IVIDTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.criterion = nn.L1Loss()
        self.metrics = MetricsTop("regression").get_metrics(config.dataset_name)

    def _move_batch(self, batch):
        return {key: value.to(device) for key, value in batch.items()}

    def _model_outputs(self, model, batch):
        return model(
            batch["text_tokens"],
            batch["text_masks"],
            batch["text_context_tokens"],
            batch["text_context_masks"],
            batch["audio_inputs"],
            batch["audio_masks"],
            batch["audio_context_inputs"],
            batch["audio_context_masks"],
            batch["video_inputs"],
            batch["video_masks"],
            batch["video_context_inputs"],
            batch["video_context_masks"],
        )

    def _loss(self, outputs, targets):
        task_loss = self.criterion(outputs["prediction"], targets)
        total_loss = (
            task_loss
            + self.config.pure_weight * outputs["L_pure"]
            + self.config.bias_weight * outputs["L_bias"]
            + self.config.complete_weight * outputs["L_complete"]
        )
        return total_loss

    def train_epoch(self, model, loader, optimizer, epoch):
        model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc=f"train {epoch}", leave=False)
        for batch in progress:
            batch = self._move_batch(batch)
            targets = batch["targets"].view(-1, 1)

            optimizer.zero_grad()
            outputs = self._model_outputs(model, batch)
            loss = self._loss(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(loader.dataset)

    def evaluate(self, model, loader, split):
        model.eval()
        total_loss = 0.0
        predictions = []
        targets_all = []

        with torch.no_grad():
            progress = tqdm(loader, desc=split, leave=False)
            for batch in progress:
                batch = self._move_batch(batch)
                targets = batch["targets"].view(-1, 1)
                outputs = self._model_outputs(model, batch)
                loss = self._loss(outputs, targets)

                total_loss += loss.item() * targets.size(0)
                predictions.append(outputs["prediction"].cpu())
                targets_all.append(targets.cpu())

        predictions = torch.cat(predictions)
        targets_all = torch.cat(targets_all)
        results = self.metrics(predictions, targets_all)
        results["Loss"] = round(total_loss / len(loader.dataset), 4)
        return results


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def run_ivid(config):
    dataset_name = config.dataset_name.lower()
    set_seed(config.seed)
    logger = build_logger(config.log, config.seed)

    logger.info("Preparing data loaders")
    train_loader, test_loader, val_loader = data_loader(
        config.batch_size,
        dataset_name,
        text_context_length=config.text_context_len,
        audio_context_length=config.audio_context_len,
    )

    logger.info("Building IVID model")
    model = IVIDModel(config).to(device)
    model.freeze_audio_feature_extractors()

    trainer = IVIDTrainer(config, logger)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-5,
    )

    checkpoint_path = (
        Path(config.checkpoint_dir) / f"ivid_{dataset_name}_seed{config.seed}.pt"
    )
    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(
        "Start training: dataset=%s batch_size=%d lr=%.2e epochs=%d patience=%d",
        dataset_name,
        config.batch_size,
        config.learning_rate,
        config.epochs,
        config.patience,
    )

    for epoch in range(1, config.epochs + 1):
        train_loss = trainer.train_epoch(model, train_loader, optimizer, epoch)
        val_results = trainer.evaluate(model, val_loader, "val")
        val_loss = val_results["Loss"]

        logger.info(
            "Epoch %03d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f | val_mae=%.4f | val_corr=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_results["Acc"],
            val_results["F1"],
            val_results["MAE"],
            val_results["Corr"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            if config.save_model:
                save_checkpoint(model, checkpoint_path)
            else:
                best_state = copy.deepcopy(model.state_dict())
            logger.info("New best validation loss at epoch %d", epoch)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(
                    "Early stopping at epoch %d; best validation epoch was %d",
                    epoch,
                    best_epoch,
                )
                break

    if config.save_model and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif best_state is not None:
        model.load_state_dict(best_state)

    test_results = trainer.evaluate(model, test_loader, "test")
    logger.info("Best epoch: %d", best_epoch)
    logger.info("Test results: %s", dict_to_str(test_results))
    return test_results

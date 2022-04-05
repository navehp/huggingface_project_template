import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    Trainer,
    EvalPrediction
)
from trainer import CustomTrainer
from datasets import load_metric

from consts import *


def get_model_obj(model_type):
    if model_type == CLS:
        return AutoModelForSequenceClassification
    elif model_type == QA:
        return AutoModelForTokenClassification
    elif model_type == TCLS:
        return AutoModelForQuestionAnswering
    else:
        raise ValueError(f"Model type {model_type} is not supported. Available types are {ALL_MODEL_TYPES}")


def get_compute_metrics(metrics):
    # Get the metric functions
    metrics = {metric: load_metric(metric) for metric in metrics}

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = {metric: metric_fn.compute(predictions=preds, references=p.label_ids)[metric] for metric, metric_fn in metrics.items()}
        return result

    return compute_metrics


def get_trainer(trainer_type):
    if trainer_type == STANDARD:
        return Trainer
    elif trainer_type == CUSTOM:
        return CustomTrainer
    else:
        raise ValueError(f"Trainer type {trainer_type} is not supported. Available types are {ALL_TRAINER_TYPES}")

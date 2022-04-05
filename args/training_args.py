from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

from consts import *


@dataclass
class ProjectTrainingArguments(TrainingArguments):
    model_type: str = field(
        default=CLS,
        metadata={"help": "Type of model head: Sequence Classification (cls), Question Answering (qa) or "
                          "Token Classification (tcls).",
                  "choices": ALL_MODEL_TYPES}
    )
    trainer_type: str = field(
        default=STANDARD,
        metadata={"help": "Type of trainer: standard huggingface trainer (standard), or custom trainer (custom)",
                  "choices": ALL_TRAINER_TYPES}
    )
    metrics: str = field(
        default='',
        metadata={"help": "Names of huggingface metrics you would like to compute seperated by commas (e.g accuracy,f1). "
                          "Notice that by default the metrics receives only predictions and labels, "
                          "for specific behavior refer to utils.train_utils.get_compute_metrics."}
    )
    return_embedding: Optional[bool] = field(
        default=False,
        metadata={"help": "Return embedding in predict loop."}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.metrics is not None:
            self.metrics = self.metrics.split(',')

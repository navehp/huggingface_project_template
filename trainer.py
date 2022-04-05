import collections
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple

from tqdm.auto import tqdm

from transformers import Trainer

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from consts import *

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
        DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging


_is_torch_generator_available = False
_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
else:
    import torch.distributed as dist

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class PredictionOutputWithOutputs(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    embeddings: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    num_samples: Optional[int]


class EvalPredictionWithGoldLabels(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray
    gold_labels_ids: np.ndarray


class CustomTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kl_loss_func = nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False, return_embeddings=False):
        target_p = inputs["labels"]
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=return_embeddings)
        logits = outputs[0]
        loss = self.kl_loss_func(logits.log_softmax(dim=-1), target_p)  # todo add case where target p isnt a dist (add softmax)

        if return_embeddings:
            last_layer_hidden_state = outputs['hidden_states'][-1]

            # Multiplying with the attention mask to zero out the pads' embeddings
            last_layer_hidden_state = (inputs['attention_mask'] * last_layer_hidden_state.permute([2, 0, 1])).permute([1, 2, 0])

            # Dividing each input with its length to get the mean
            lengths = inputs['attention_mask'].sum(dim=1)
            embeddings = (last_layer_hidden_state.sum(dim=1).t() / lengths).t()
            return loss, outputs, embeddings

        if return_outputs:
            return loss, outputs

        return loss

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        gold_labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_gold_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs[GOLD_LABELS] is not None:
                gold_labels = self._pad_across_processes(inputs[GOLD_LABELS])
                gold_labels = self._nested_gather(gold_labels)
                gold_labels_host = gold_labels if gold_labels_host is None else nested_concat(gold_labels_host, gold_labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                if gold_labels_host is not None:
                    gold_labels = nested_numpify(gold_labels_host)
                    all_gold_labels = (
                        gold_labels if all_gold_labels is None else nested_concat(all_gold_labels, gold_labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if gold_labels_host is not None:
            gold_labels = nested_numpify(gold_labels_host)
            all_gold_labels = gold_labels if all_gold_labels is None else nested_concat(all_gold_labels, gold_labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_gold_labels is not None:
            all_gold_labels = nested_truncate(all_gold_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None and all_gold_labels is not None:
            metrics = self.compute_metrics(EvalPredictionWithGoldLabels(predictions=all_preds, label_ids=all_labels, gold_labels_ids=all_gold_labels))
        elif self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    # def prediction_loop(
    #     self,
    #     dataloader: DataLoader,
    #     description: str,
    #     prediction_loss_only: Optional[bool] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> PredictionOutput:
    #     """
    #     Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.
    #
    #     Works both with or without labels.
    #     """
    #     if not isinstance(dataloader.dataset, collections.abc.Sized):
    #         raise ValueError("dataset must implement __len__")
    #     prediction_loss_only = (
    #         prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
    #     )
    #
    #     model = self._wrap_model(self.model, training=False)
    #
    #     # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
    #     # ``train`` is running, halve it first and then put on device
    #     if not self.is_in_train and self.args.fp16_full_eval:
    #         model = model.half().to(self.args.device)
    #
    #     batch_size = dataloader.batch_size
    #     num_examples = self.num_examples(dataloader)
    #     logger.info(f"***** Running {description} *****")
    #     logger.info(f"  Num examples = {num_examples}")
    #     logger.info(f"  Batch size = {batch_size}")
    #     losses_host: torch.Tensor = None
    #     preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
    #     labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
    #     gold_labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
    #
    #     world_size = max(1, self.args.world_size)
    #
    #     eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
    #     if not prediction_loss_only:
    #         # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
    #         # a batch size to the sampler)
    #         make_multiple_of = None
    #         if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, SequentialDistributedSampler):
    #             make_multiple_of = dataloader.sampler.batch_size
    #         preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
    #         labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
    #         gold_labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
    #
    #     model.eval()
    #
    #     if self.args.past_index >= 0:
    #         self._past = None
    #
    #     self.callback_handler.eval_dataloader = dataloader
    #
    #     for step, inputs in enumerate(dataloader):
    #         loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
    #         if loss is not None:
    #             losses = loss.repeat(batch_size)
    #             losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
    #         if logits is not None:
    #             preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
    #         if labels is not None:
    #             labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
    #             gold_labels_host = inputs[GOLD_LABELS] if gold_labels_host is None else nested_concat(gold_labels_host, inputs[GOLD_LABELS], padding_index=-100)
    #         self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)
    #
    #         # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
    #         if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
    #             eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
    #             if not prediction_loss_only:
    #                 preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
    #                 labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
    #                 gold_labels_gatherer.add_arrays(self._gather_and_numpify(gold_labels_host, "eval_gold_label_ids"))
    #
    #             # Set back to None to begin a new accumulation
    #             losses_host, preds_host, labels_host, gold_labels_host = None, None, None, None
    #
    #     if self.args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of the evaluation loop
    #         delattr(self, "_past")
    #
    #     # Gather all remaining tensors and put them back on the CPU
    #     eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
    #     if not prediction_loss_only:
    #         preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
    #         labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
    #         gold_labels_gatherer.add_arrays(self._gather_and_numpify(gold_labels_host, "eval_label_ids"))
    #
    #     eval_loss = eval_losses_gatherer.finalize()
    #     preds = preds_gatherer.finalize() if not prediction_loss_only else None
    #     label_ids = labels_gatherer.finalize() if not prediction_loss_only else None
    #     gold_label_ids = gold_labels_gatherer.finalize() if not prediction_loss_only else None
    #
    #     # Number of samples
    #     eval_dataset = dataloader.dataset
    #     if not isinstance(eval_dataset, IterableDataset):
    #         num_samples = len(eval_dataset)
    #     # The instance check is weird and does not actually check for the type, but whether the dataset has the right
    #     # methods. Therefore we need to make sure it also has the attribute.
    #     elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
    #         num_samples = eval_dataset.num_examples
    #     else:
    #         num_samples = observed_num_examples
    #
    #     if self.compute_metrics is not None and preds is not None and label_ids is not None and gold_label_ids is not None:
    #         metrics = self.compute_metrics(EvalPredictionWithGoldLabels(predictions=preds, label_ids=label_ids, gold_labels_ids=gold_label_ids))
    #     elif self.compute_metrics is not None and preds is not None and label_ids is not None:
    #         metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
    #     else:
    #         metrics = {}
    #
    #     # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    #     metrics = denumpify_detensorize(metrics)
    #
    #     if eval_loss is not None:
    #         metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()
    #
    #     # Prefix all keys with metric_key_prefix + '_'
    #     for key in list(metrics.keys()):
    #         if not key.startswith(f"{metric_key_prefix}_"):
    #             metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
    #
    #     return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    # Copied the original predict method and added embeddings
    def predict_and_embed(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test",
            return_embeddings: bool = True
    ) -> Union[PredictionOutput, PredictionOutputWithOutputs]:
        """
        Run prediction and returns predictions, sequence embeddings and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        .. note::

            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """

        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.prediction_and_embedding_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, return_embeddings=return_embeddings
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    # Copied the original prediction_step from Trainer, add return_outputs option
    def prediction_and_embedding_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        return_embeddings: Optional[bool] = False
    ) -> Union[
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
    ]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                if return_embeddings:
                    loss, outputs, embeddings = self.compute_loss(model, inputs, return_outputs=True, return_embeddings=True)
                else:
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict) and return_embeddings:
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss", "hidden_states"])
                elif isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if return_embeddings:
            return (loss, logits, labels, embeddings)
        return (loss, logits, labels)


    def prediction_and_embedding_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        return_embeddings: Optional[bool] = True,
    ) -> Union[PredictionOutput, PredictionOutputWithOutputs]:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        embeddings_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = max(1, self.args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, SequentialDistributedSampler):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            if return_embeddings:
                embeddings_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            if return_embeddings:
                loss, logits, labels, embeddings = self.prediction_and_embedding_step(model, inputs, prediction_loss_only,
                                                            ignore_keys=ignore_keys, return_embeddings=True)
            else:
                loss, logits, labels = self.prediction_and_embedding_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if return_embeddings and embeddings is not None:
                embeddings_host = embeddings if embeddings_host is None else nested_concat(embeddings_host, embeddings, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
                    if return_embeddings:
                        embeddings_gatherer.add_arrays(self._gather_and_numpify(embeddings_host, "eval_outputs"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None
                if return_embeddings:
                    embeddings_host = None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
            if return_embeddings:
                embeddings_gatherer.add_arrays(self._gather_and_numpify(embeddings_host, "eval_outputs"))


        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None
        embeddings = embeddings_gatherer.finalize() if return_embeddings and not prediction_loss_only else None

        eval_dataset = dataloader.dataset
        num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        if return_embeddings:
            return PredictionOutputWithOutputs(predictions=preds, label_ids=label_ids, metrics=metrics, embeddings=embeddings, num_samples=num_samples)
        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
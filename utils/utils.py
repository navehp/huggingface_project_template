import pickle
import json
import os
import shutil
import sys
import logging
import importlib.util

import pandas as pd
from transformers import utils
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from datasets import Dataset, DatasetDict, load_metric

from consts import *


def setup_logging(data_args, model_args, training_args, logger):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    utils.logging.set_verbosity(log_level)
    utils.logging.set_verbosity(log_level)
    utils.logging.enable_default_handler()
    utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    if training_args.local_rank != -1:
        raise ValueError("Distributed training is not currently supported.")
    if training_args.tpu_num_cores is not None:
        raise ValueError("TPU acceleration is not currently supported.")

    logger.info(f"Training args {training_args}")
    logger.info(f"Data args {data_args}")
    logger.info(f"Teacher args {model_args}")  # todo have better printing of args

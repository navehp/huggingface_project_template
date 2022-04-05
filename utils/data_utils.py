import os
from pathlib import Path

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

from consts import *


def load_glue(data_args, suffix='.json'):  # todo add this to consts
    data_dir = DATA_DIR / data_args.dataset / data_args.task_name
    data_files = {
        TRAIN: data_dir / (TRAIN + suffix),
        VALIDATION: data_dir / (VALIDATION + suffix),
        TEST: data_dir / (TEST + suffix)
    }
    raw_datasets = load_dataset(suffix, data_files=data_files)
    return raw_datasets
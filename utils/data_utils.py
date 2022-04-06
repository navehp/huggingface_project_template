from datasets import load_dataset

from consts import *


def load_dataset_from_files(data_args, suffix=JSON):
    data_dir = DATA_DIR / data_args.dataset
    data_files = {
        TRAIN: data_dir / (TRAIN + suffix),
        VALIDATION: data_dir / (VALIDATION + suffix),
        TEST: data_dir / (TEST + suffix)
    }
    raw_datasets = load_dataset(suffix, data_files=data_files)
    return raw_datasets
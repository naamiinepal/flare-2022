import os
import warnings

import torch

from Common.gpu_utils import set_gpu
from BaseSeg.config.config import get_cfg_defaults

warnings.filterwarnings("ignore")


def get_configs():
    CONFIG_FILE = "/home/safal/flare/FlareSeg/coarse_base_seg/config.yaml"
    LOCAL_RANK = 0

    cfg = get_cfg_defaults()
    cfg.merge_from_file(CONFIG_FILE)
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    cfg.ENVIRONMENT.DATA_BASE_DIR = os.path.join(
        base_dir, cfg.ENVIRONMENT.DATA_BASE_DIR
    )
    cfg.DATA_LOADER.BAD_CASE_SERIES_IDS_TXT = os.path.join(
        base_dir, cfg.DATA_LOADER.BAD_CASE_SERIES_IDS_TXT
    )

    set_gpu(cfg.ENVIRONMENT.NUM_GPU, used_percent=0.2, local_rank=int(LOCAL_RANK))
    if cfg.ENVIRONMENT.CUDA and torch.cuda.is_available():
        cfg.ENVIRONMENT.CUDA = True
        torch.cuda.manual_seed_all(cfg.ENVIRONMENT.SEED)
    else:
        cfg.ENVIRONMENT.CUDA = False
        torch.manual_seed(cfg.ENVIRONMENT.SEED)

    if cfg.TRAINING.IS_DISTRIBUTED_TRAIN and cfg.ENVIRONMENT.NUM_GPU > 1:
        cfg.TRAINING.IS_DISTRIBUTED_TRAIN = True
    else:
        cfg.TRAINING.IS_DISTRIBUTED_TRAIN = False

    if cfg.ENVIRONMENT.DATA_BASE_DIR is not None:
        cfg.DATA_LOADER.TRAIN_DB_FILE = (
            cfg.ENVIRONMENT.DATA_BASE_DIR + cfg.DATA_LOADER.TRAIN_DB_FILE
        )
        cfg.DATA_LOADER.VAL_DB_FILE = (
            cfg.ENVIRONMENT.DATA_BASE_DIR + cfg.DATA_LOADER.VAL_DB_FILE
        )

    cfg.freeze()
    return cfg

from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_thread_env() -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
    os.environ.setdefault("OMP_NUM_THREADS", "64")
    os.environ.setdefault("MKL_NUM_THREADS", "64")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

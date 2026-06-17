from __future__ import annotations

from ..stage_handler_common import BaseTrainingStageHandler
from .losses import IonizationLoss


class TrainingStageHandler(BaseTrainingStageHandler):
    loss_cls = IonizationLoss

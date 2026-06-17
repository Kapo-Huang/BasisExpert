from __future__ import annotations

from ..stage_handler_common import BaseTrainingStageHandler
from .losses import MeshReconstructionLoss


class TrainingStageHandler(BaseTrainingStageHandler):
    loss_cls = MeshReconstructionLoss

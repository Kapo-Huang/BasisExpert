from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol

import numpy as np

if TYPE_CHECKING:
    from .service import EvaluationRequest


class DecodeSession(Protocol):
    """Model-independent selected-frame decoder contract."""

    source_kind: str
    source_path: Path
    selection_mode: str

    def frames(self) -> Iterator[tuple[str, int, np.ndarray]]: ...


class RunAdapter(Protocol):
    """Contract for resolving and evaluating a saved model run."""

    name: str

    def evaluate(
        self,
        request: "EvaluationRequest",
        raw: dict[str, Any],
        config_path: Path,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class StandardRunAdapter:
    name: str = "unified"

    def evaluate(self, request, raw, config_path):
        from .service import run_standard_evaluation

        return run_standard_evaluation(request)


@dataclass(frozen=True)
class StandaloneRunAdapter:
    subsystem: str

    @property
    def name(self) -> str:
        return self.subsystem

    def evaluate(self, request, raw, config_path):
        from .standalone import run_standalone_evaluation

        return run_standalone_evaluation(request, raw, self.subsystem, config_path)


def select_run_adapter(raw: dict[str, Any]) -> RunAdapter:
    from .standalone import identify_subsystem

    subsystem = identify_subsystem(raw)
    if subsystem is not None:
        return StandaloneRunAdapter(subsystem)
    if "model" in raw:
        return StandardRunAdapter()
    raise ValueError("Unable to identify the run model/subsystem from its saved config")


SUPPORTED_ADAPTERS = (
    "unified",
    "compact_ngp",
    "mc_inr",
    "dc_inr",
    "apmgsrn",
    "fv_srn",
    "rmdsrn",
    "ecnr",
    "neural_expert",
)

from __future__ import annotations


class _DummyRun:
    def __init__(self, project: str):
        self.id = "disabled"
        self.project = project
        self.entity = "disabled"
        self.name = ""

    def log_code(self, *args, **kwargs):
        return None


class _DummyConfig:
    def update(self, *args, **kwargs):
        return None


class _DummyWandb:
    def __init__(self):
        self.run = _DummyRun(project="disabled")
        self.config = _DummyConfig()

    def init(self, project=None, entity=None, save_code=None, dir=None, mode=None):
        self.run = _DummyRun(project=project or "disabled")
        return self.run

    def define_metric(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None


def get_wandb():
    try:
        import wandb
    except Exception:
        return _DummyWandb()
    return wandb

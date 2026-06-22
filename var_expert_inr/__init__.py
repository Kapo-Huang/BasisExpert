from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "var_expert_inr"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))

try:
    from .utils.runtime import configure_thread_env
except Exception:
    configure_thread_env = None
else:
    configure_thread_env()

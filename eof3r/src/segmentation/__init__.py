# EOF3R segmentation module — scene decomposition via SAM2

try:
    from .sam2_wrapper import SAM2Wrapper
except ImportError:
    SAM2Wrapper = None  # type: ignore[assignment]

from .sam2_stub import SAM2Stub

# Prefer real wrapper, fall back to stub.
SAM2 = SAM2Wrapper if SAM2Wrapper is not None else SAM2Stub

__all__ = ["SAM2", "SAM2Stub"]
if SAM2Wrapper is not None:
    __all__.append("SAM2Wrapper")

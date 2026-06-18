# EOF3R background module — 3R coarse geometry estimation via VGGT

try:
    from .vggt_wrapper import VGGTWrapper
except ImportError:
    VGGTWrapper = None  # type: ignore[assignment]

from .vggt_stub import VGGTStub

# Prefer real wrapper, fall back to stub.
VGGT = VGGTWrapper if VGGTWrapper is not None else VGGTStub

__all__ = ["VGGT", "VGGTStub"]
if VGGTWrapper is not None:
    __all__.append("VGGTWrapper")

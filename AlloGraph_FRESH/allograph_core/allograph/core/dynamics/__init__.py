from .diffusion import DiffusionModel
from .discrete import DiscreteHopModel

def make_model(name: str):
    name = (name or "").lower().strip()
    if name in ("diffusion", "laplacian", "heat"):
        return DiffusionModel()
    if name in ("discrete", "hop"):
        return DiscreteHopModel()
    raise ValueError(f"Unknown model: {name}")

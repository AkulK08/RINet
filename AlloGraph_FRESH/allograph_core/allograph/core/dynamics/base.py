from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

@dataclass(frozen=True)
class DynamicsResult:
    """
    Immutable container for dynamics output.
    
    - state: The final propagated state vector.
    - history: Optional list of states over time.
    - meta: Metadata such as parameters used.
    """
    state: np.ndarray
    meta: Dict[str, Any]
    history: Optional[list[np.ndarray]] = field(default=None)


class BaseDynamicsModel:
    """
    Abstract base class for all dynamics models.

    Subclasses should override the run() method.
    The name attribute identifies the model in metadata.
    """
    name: str = "base"

    def run(
        self,
        A: np.ndarray,
        x0: np.ndarray,
        steps: int,
        dt: float,
        **kwargs: Any
    ) -> DynamicsResult:
        """
        Execute the dynamics model.

        Parameters
        ----------
        A : adjacency or interaction matrix
        x0 : initial state
        steps : number of evolution steps
        dt : time increment
        **kwargs : model-specific parameters (e.g., alpha, history flag)

        Returns
        -------
        DynamicsResult
        """
        raise NotImplementedError("Subclasses must implement run()")

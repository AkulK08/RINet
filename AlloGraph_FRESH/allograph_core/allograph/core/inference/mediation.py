from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict

from ..graphio.types import GraphBundle

@dataclass
class MediationResult:
    centrality: np.ndarray
    meta: Dict[str, Any]


def run_mediation(
    bundle: GraphBundle,
    method: str = "betweenness",
    normalized: bool = True,
) -> MediationResult:
    """
    Compute a mediation-oriented centrality measure for the graph.

    Args
    ----
    bundle
        A GraphBundle representing the adjacency and metadata.
    method
        Which centrality measure to compute:
            - "betweenness": betweenness centrality
            - "closeness": closeness centrality
            - "eigenvector": eigenvector centrality
    normalized
        Whether to normalize centrality scores
        (standard practice in centrality metrics).

    Returns
    -------
    MediationResult with centrality vector and metadata.
    
    Notes
    -----
    - Betweenness is often interpreted as a mediation/broker role
      because it quantifies how often a node lies on shortest paths
      between all other pairs of nodes. :contentReference[oaicite:7]{index=7}
    - Closeness quantifies how centrally located a node is by
      distance. :contentReference[oaicite:8]{index=8}
    - Eigenvector captures influence in global connectivity. :contentReference[oaicite:9]{index=9}
    """

    bundle.validate()
    A = bundle.A
    n = A.shape[0]

    if method == "betweenness":
        # Compute all-pairs shortest paths and count mediating occurrences
        # Brandes' algorithm is the standard deterministic approach. :contentReference[oaicite:10]{index=10}

        # Convert adjacency to unweighted graph for shortest paths
        # (works for weighted if desired)
        graph = {i: list(np.where(A[i] > 0)[0]) for i in range(n)}

        # Brandes' betweenness algorithm (deterministic)
        centrality = np.zeros(n, dtype=float)
        for s in range(n):
            stack: list[int] = []
            preds: list[list[int]] = [[] for _ in range(n)]
            sigma = np.zeros(n, dtype=float)
            dist = -np.ones(n, dtype=float)

            sigma[s] = 1.0
            dist[s] = 0
            queue = [s]

            # Breadth-first search
            while queue:
                v = queue.pop(0)
                stack.append(v)
                for w in graph[v]:
                    # Path discovery
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    # Path counting
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        preds[w].append(v)

            # Accumulate dependencies
            delta = np.zeros(n, dtype=float)
            while stack:
                w = stack.pop()
                for v in preds[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    centrality[w] += delta[w]

        if normalized and n > 2:
            # Normalization factor for undirected graphs:
            # divide by (n-1)(n-2)/2 to scale centrality into [0,1].
            centrality /= ((n - 1) * (n - 2) / 2)

    elif method == "closeness":
        # Closeness centrality
        # Inverse of sum of shortest distances from node to others. :contentReference[oaicite:11]{index=11}
        from collections import deque

        centrality = np.zeros(n, dtype=float)
        for i in range(n):
            dist = -np.ones(n, dtype=float)
            dist[i] = 0
            queue = deque([i])
            while queue:
                v = queue.popleft()
                for w in range(n):
                    if A[v, w] > 0 and dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
            # sum of distances to all reachable nodes
            reachable = dist[dist >= 0]
            if reachable.size > 1:
                centrality[i] = (reachable.size - 1) / np.sum(reachable[reachable > 0])
            else:
                centrality[i] = 0.0

    elif method == "eigenvector":
        # Eigenvector centrality
        # Solve x = λ A x (principal eigenvector). :contentReference[oaicite:12]{index=12}
        # Use power iteration
        x = np.ones(n, dtype=float)
        for _ in range(100):
            x_next = A @ x
            norm = np.linalg.norm(x_next)
            if norm == 0:
                break
            x = x_next / norm
        centrality = x

    else:
        raise ValueError(f"Unknown centrality method '{method}'")

    meta: Dict[str, Any] = dict(bundle.meta)
    meta.update({"mediation_method": method, "normalized": bool(normalized)})

    return MediationResult(centrality=centrality, meta=meta)

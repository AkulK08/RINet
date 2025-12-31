from __future__ import annotations

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import QWidget, QVBoxLayout

import networkx as nx


def _to_dense(A):
    # Handles numpy arrays and scipy sparse (if present) without importing scipy explicitly
    if hasattr(A, "toarray"):
        return np.asarray(A.toarray(), dtype=float)
    return np.asarray(A, dtype=float)


class PlotPanel(QWidget):
    """
    Matplotlib canvas with two plotting modes:
      - plot_xy(y): classic residue-index signal plot
      - plot_network(bundle, values): node-link network view (edges weighted, nodes colored by values)
    """
    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clear(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.canvas.draw_idle()

    def plot_xy(self, y, title="Signal", xlabel="Residue index", ylabel="Value"):
        y = np.asarray(y, dtype=float).ravel()
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.plot(np.arange(len(y)), y)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def plot_network(
        self,
        bundle,
        values=None,
        title="RIN network view",
        max_edges=4000,
        layout_mode="spring",
        seed=0,
        edge_gamma=0.8,
        node_size=55,
    ):
        """
        bundle.A: adjacency (dense or sparse)
        values: length-N vector for node coloring (forward/scan/inverse scores). Optional.
        Edge thickness reflects weight; if binary, all edges same thickness.
        For dense graphs, edges are downsampled to max_edges.
        """
        A = _to_dense(bundle.A)
        n = int(getattr(bundle, "n", A.shape[0]))

        # Build list of edges from upper triangle
        iu, ju = np.triu_indices(n, k=1)
        w = A[iu, ju]
        mask = w > 0
        i = iu[mask]
        j = ju[mask]
        w = w[mask]

        # Downsample edges if too many
        m = len(w)
        if m > max_edges:
            # keep strongest edges
            idx = np.argsort(w)[-max_edges:]
            i, j, w = i[idx], j[idx], w[idx]

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for a, b, ww in zip(i.tolist(), j.tolist(), w.tolist()):
            G.add_edge(a, b, weight=float(ww))

        # Layout
        if layout_mode == "kamada":
            pos = nx.kamada_kawai_layout(G)
        elif layout_mode == "spectral":
            pos = nx.spectral_layout(G)
        else:
            # spring is usually nicest
            pos = nx.spring_layout(G, seed=seed)

        # Prepare node colors
        if values is None:
            node_c = None
        else:
            v = np.asarray(values, dtype=float).ravel()
            if len(v) != n:
                raise ValueError(f"values length {len(v)} != n {n}")
            # normalize robustly
            vmin = np.percentile(v, 2)
            vmax = np.percentile(v, 98)
            if vmax <= vmin:
                vmax = vmin + 1e-9
            node_c = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)

        # Edge widths from weights
        if len(w) > 0:
            wv = np.asarray(w, dtype=float)
            # normalize weights
            wmin = float(np.min(wv))
            wmax = float(np.max(wv))
            if wmax <= wmin:
                widths = np.full_like(wv, 1.5, dtype=float)
                alphas = np.full_like(wv, 0.35, dtype=float)
            else:
                t = (wv - wmin) / (wmax - wmin)
                # gamma makes contrast nicer
                t = np.power(t, edge_gamma)
                widths = 0.3 + 4.0 * t
                alphas = 0.08 + 0.42 * t
        else:
            widths = []
            alphas = []

        # Plot
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.set_title(title)
        ax.axis("off")

        # draw edges individually so alpha can vary
        edges = list(G.edges())
        if edges:
            for (e, width, alpha) in zip(edges, widths, alphas):
                x0, y0 = pos[e[0]]
                x1, y1 = pos[e[1]]
                ax.plot([x0, x1], [y0, y1], linewidth=float(width), alpha=float(alpha))

        # draw nodes
        xs = np.array([pos[k][0] for k in range(n)], dtype=float)
        ys = np.array([pos[k][1] for k in range(n)], dtype=float)

        if node_c is None:
            ax.scatter(xs, ys, s=node_size)
        else:
            ax.scatter(xs, ys, s=node_size, c=node_c, cmap="viridis")

        self.fig.tight_layout()
        self.canvas.draw_idle()

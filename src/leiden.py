"""
leiden_cora.py
==============
Runs Leiden on a synthetic Cora-like graph and plots the result.
200 nodes, 7 ground-truth classes.

Libraries:
  - igraph    : graph data structure (required by leidenalg)
  - leidenalg : Leiden algorithm
  - matplotlib: plotting
"""

import collections
import random
from collections import defaultdict

import igraph as ig
import leidenalg
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

random.seed(42)

# -----------------------------------------------------------------------
# SECTION 1 — BUILD SYNTHETIC CORA-LIKE GRAPH (200 nodes)
# -----------------------------------------------------------------------
N_NODES   = 200
N_CLASSES = 7
INTRA_P   = 0.12    # high within-class edge probability
INTER_P   = 0.005   # low between-class edge probability

labels = []
base = N_NODES // N_CLASSES
for c in range(N_CLASSES):
    count = base if c < N_CLASSES - 1 else N_NODES - base * (N_CLASSES - 1)
    labels.extend([c] * count)

edges = []
for i in range(N_NODES):
    for j in range(i + 1, N_NODES):
        p = INTRA_P if labels[i] == labels[j] else INTER_P
        if random.random() < p:
            edges.append((i, j))

# -----------------------------------------------------------------------
# SECTION 2 — BUILD igraph.Graph
# -----------------------------------------------------------------------
# ig.Graph(n, edges, directed)
#   n     : number of vertices
#   edges : list of (src, dst) integer index tuples
# .vs["label"] attaches ground-truth class as a vertex attribute
# .simplify() removes self-loops and duplicate edges

G = ig.Graph(n=N_NODES, edges=edges, directed=False)
G.vs["label"] = labels
G.simplify()

# -----------------------------------------------------------------------
# SECTION 3 — RUN LEIDEN
# -----------------------------------------------------------------------
# find_partition(graph, partition_type, n_iterations, seed)
#   ModularityVertexPartition : optimises modularity Q
#   n_iterations=-1           : run until convergence
#
# Returns VertexPartition:
#   .membership  : List[int] of length N_NODES — community ID per node
#   .modularity  : float — quality score Q

partition = leidenalg.find_partition(
    G,
    leidenalg.ModularityVertexPartition,
    n_iterations=-1,
    seed=42,
)

membership = partition.membership
n_comm     = len(set(membership))
comm_sizes = collections.Counter(membership)

# -----------------------------------------------------------------------
# SECTION 4 — PURITY SCORE
# -----------------------------------------------------------------------
# Purity = (1/N) * sum_k  max_j |c_k ∩ g_j|
# For each community, count nodes belonging to the majority ground-truth class.

comm_to_labels = defaultdict(list)
for node_id, comm_id in enumerate(membership):
    comm_to_labels[comm_id].append(labels[node_id])

total_correct = sum(
    collections.Counter(node_labels).most_common(1)[0][1]
    for node_labels in comm_to_labels.values()
)
purity = total_correct / N_NODES

print(f"Nodes       : {G.vcount()}")
print(f"Edges       : {G.ecount()}")
print(f"Communities : {n_comm}")
print(f"Modularity  : {partition.modularity:.4f}")
print(f"Purity      : {purity:.4f}")

# -----------------------------------------------------------------------
# SECTION 5 — PLOT
# -----------------------------------------------------------------------
# Layout: Fruchterman-Reingold (force-directed)
#   Nodes repel each other, edges act as springs.
#   Nodes with many shared edges end up close together.
#   This is purely structural — no label/community info used.

layout = G.layout("fr", niter=1000)
coords = layout.coords   # list of [x, y] per node

COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabebe", "#469990", "#dcbeff", "#9A6324",
    "#fffac8", "#800000", "#aaffc3",
]

xs = [c[0] for c in coords]
ys = [c[1] for c in coords]

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f0f0f")

for ax in axes:
    ax.set_facecolor("#0f0f0f")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# --- LEFT: ground-truth labels ---
gt_colors = [COLORS[labels[i] % len(COLORS)] for i in range(N_NODES)]

for e in G.get_edgelist():
    x0, y0 = coords[e[0]]
    x1, y1 = coords[e[1]]
    axes[0].plot([x0, x1], [y0, y1], color="#444444", lw=0.4, alpha=0.5, zorder=1)

axes[0].scatter(xs, ys, c=gt_colors, s=40, zorder=2,
                edgecolors="#111111", linewidths=0.3)
axes[0].set_title("Ground Truth Labels (7 classes)",
                  color="white", fontsize=13, pad=10)

gt_patches = [mpatches.Patch(color=COLORS[c], label=f"Class {c}")
              for c in range(N_CLASSES)]
axes[0].legend(handles=gt_patches, loc="upper left", framealpha=0.2,
               labelcolor="white", fontsize=8,
               facecolor="#1a1a1a", edgecolor="#444444")

# --- RIGHT: Leiden communities ---
sorted_comms  = [c for c, _ in sorted(comm_sizes.items(), key=lambda x: -x[1])]
comm_rank     = {c: i for i, c in enumerate(sorted_comms)}
leiden_colors = [COLORS[comm_rank[membership[i]] % len(COLORS)]
                 for i in range(N_NODES)]

for e in G.get_edgelist():
    x0, y0 = coords[e[0]]
    x1, y1 = coords[e[1]]
    axes[1].plot([x0, x1], [y0, y1], color="#444444", lw=0.4, alpha=0.5, zorder=1)

axes[1].scatter(xs, ys, c=leiden_colors, s=40, zorder=2,
                edgecolors="#111111", linewidths=0.3)
axes[1].set_title(
    f"Leiden Communities ({n_comm} detected) | Q={partition.modularity:.3f}",
    color="white", fontsize=13, pad=10
)

leiden_patches = [
    mpatches.Patch(
        color=COLORS[comm_rank[c] % len(COLORS)],
        label=f"Comm {c} (n={comm_sizes[c]})"
    )
    for c in sorted_comms if comm_sizes[c] > 2
]
axes[1].legend(handles=leiden_patches, loc="upper left", framealpha=0.2,
               labelcolor="white", fontsize=8,
               facecolor="#1a1a1a", edgecolor="#444444")

plt.suptitle("Leiden Community Detection on Synthetic Cora (200 nodes)",
             color="white", fontsize=15, y=1.01)
plt.tight_layout()
plt.show()
# plt.savefig("leiden_cora.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
# plt.close()
# print("Plot saved to leiden_cora.png")

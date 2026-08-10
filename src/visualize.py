"""
Graph Visualization Module for Fraud Ring Detector.
Provides static (NetworkX / Matplotlib) and interactive (Plotly / PyVis) graph renders with color-coded ring subgraphs.
"""

import os
from typing import List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import networkx as nx

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def extract_fraud_ring_subgraphs(G: nx.Graph, min_ring_size: int = 2) -> List[nx.Graph]:
    """
    Extracts connected component subgraphs that contain at least one flagged fraud node.

    Args:
        G (nx.Graph): Full applicant graph.
        min_ring_size (int): Minimum nodes required to be considered a ring subgraph.

    Returns:
        List[nx.Graph]: List of extracted fraud ring subgraphs.
    """
    ring_subgraphs = []
    for component in nx.connected_components(G):
        if len(component) >= min_ring_size:
            subG = G.subgraph(component).copy()
            # Check if component contains any fraud node
            fraud_count = sum(1 for node in subG.nodes() if subG.nodes[node].get("is_fraud", 0) == 1)
            if fraud_count > 0:
                ring_subgraphs.append(subG)
    return ring_subgraphs


def plot_static_graph(G: nx.Graph, title: str = "Applicant Graph Visualization", save_path: Optional[str] = None):
    """
    Plots a static 2D graph layout with NetworkX and Matplotlib.
    Legitimate nodes = Blue (#3B82F6), Fraud nodes = Red (#EF4444).
    """
    if not HAS_MATPLOTLIB:
        print("INFO: matplotlib not installed. Skipping static graph plot.")
        return

    pos = nx.spring_layout(G, seed=42, k=0.15)
    
    node_colors = []
    node_sizes = []

    for node in G.nodes():
        is_fraud = G.nodes[node].get("is_fraud", 0)
        if is_fraud == 1:
            node_colors.append("#EF4444")  # Red
            node_sizes.append(250)
        else:
            node_colors.append("#3B82F6")  # Blue
            node_sizes.append(80)

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="#94A3B8", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85, ax=ax)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved static graph plot to '{save_path}'.")
    plt.close()


def create_interactive_plotly_graph(G: nx.Graph, title: str = "Interactive Applicant Graph Network") -> Any:
    """
    Creates an interactive 2D Plotly graph figure highlighting fraud rings.
    """
    if not HAS_PLOTLY:
        return None

    pos = nx.spring_layout(G, seed=42, k=0.2)

    # Edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#CBD5E1'),
        hoverinfo='none',
        mode='lines'
    )

    # Node traces
    node_x = []
    node_y = []
    node_colors = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        attrs = G.nodes[node]
        app_id = attrs.get("applicant_id", f"APP_{node}")
        is_fraud = attrs.get("is_fraud", 0)
        bank = attrs.get("bank_account", "N/A")
        income = attrs.get("income", 0)
        loan = attrs.get("loan_amount", 0)

        if is_fraud == 1:
            node_colors.append('#EF4444')  # Red
            status_str = "⚠️ FLAG: FRAUD RING"
        else:
            node_colors.append('#3B82F6')  # Blue
            status_str = "✅ Legitimate"

        hover_str = f"<b>{app_id}</b> ({status_str})<br>Bank: {bank}<br>Income: ${income:,.2f}<br>Loan: ${loan:,.2f}"
        node_text.append(hover_str)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colors,
            size=12,
            line=dict(width=1, color='#1E293B')
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )
    return fig

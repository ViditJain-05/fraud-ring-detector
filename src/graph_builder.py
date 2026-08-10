"""
Graph Builder Module for Fraud Ring Detector.
Constructs NetworkX loan applicant graphs based on shared attributes (bank, address, phone)
and converts them into PyTorch Geometric Data structures.
"""

import os
import sys
import csv
from typing import Tuple, Dict, Any, List, Union
import numpy as np
import networkx as nx
import torch

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.utils import from_networkx
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    PyGData = None


def load_applicants_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Loads CSV dataset into list of dictionaries."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "applicant_id": row["applicant_id"],
                "phone": row["phone"],
                "bank_account": row["bank_account"],
                "address": row["address"],
                "income": float(row["income"]),
                "loan_amount": float(row["loan_amount"]),
                "age": int(float(row["age"])),
                "city": row.get("city", "Unknown"),
                "is_fraud": int(float(row["is_fraud"]))
            })
    return records


def build_applicant_graph(data_input: Union[List[Dict[str, Any]], Any]) -> Tuple[nx.Graph, List[Dict[str, Any]]]:
    """
    Builds an undirected NetworkX graph from applicant records (DataFrame or List of dicts).
    Nodes represent applicants, edges connect applicants sharing bank_account, address, or phone.
    """
    if HAS_PANDAS and isinstance(data_input, pd.DataFrame):
        records = data_input.to_dict(orient="records")
    elif isinstance(data_input, str):
        records = load_applicants_from_csv(data_input)
    else:
        records = data_input

    G = nx.Graph()

    # Calculate normalization stats for income, loan_amount, age
    incomes = [r["income"] for r in records]
    loans = [r["loan_amount"] for r in records]
    ages = [r["age"] for r in records]

    inc_m, inc_s = np.mean(incomes), np.std(incomes) + 1e-6
    loan_m, loan_s = np.mean(loans), np.std(loans) + 1e-6
    age_m, age_s = np.mean(ages), np.std(ages) + 1e-6

    # Add Nodes
    for idx, row in enumerate(records):
        norm_inc = float((row["income"] - inc_m) / inc_s)
        norm_loan = float((row["loan_amount"] - loan_m) / loan_s)
        norm_age = float((row["age"] - age_m) / age_s)

        G.add_node(
            idx,
            applicant_id=row["applicant_id"],
            income=norm_inc,
            loan_amount=norm_loan,
            age=norm_age,
            is_fraud=int(row["is_fraud"]),
            bank_account=row["bank_account"],
            address=row["address"],
            phone=row["phone"]
        )

    # Edge Lookup via Hash Tables
    attr_cols = ["bank_account", "address", "phone"]
    for attr in attr_cols:
        groups: Dict[str, List[int]] = {}
        for idx, row in enumerate(records):
            val = str(row[attr])
            groups.setdefault(val, []).append(idx)

        for attr_val, node_indices in groups.items():
            if len(node_indices) > 1:
                for i in range(len(node_indices)):
                    for j in range(i + 1, len(node_indices)):
                        u, v = node_indices[i], node_indices[j]
                        if G.has_edge(u, v):
                            G[u][v]["shared_attributes"].append(attr)
                        else:
                            G.add_edge(u, v, shared_attributes=[attr])

    return G, records



def convert_to_pyg_data(G: nx.Graph) -> Any:
    """
    Converts a NetworkX graph into a PyTorch Geometric Data object or a fallback custom PyTorch container.

    Args:
        G (nx.Graph): Input NetworkX graph.

    Returns:
        Any: PyG Data object or custom dict container containing (x, edge_index, y).
    """
    nodes_sorted = sorted(G.nodes())

    # Extract Node Features [income, loan_amount, age] -> Shape (N, 3)
    x_list = []
    y_list = []

    for node in nodes_sorted:
        attrs = G.nodes[node]
        x_list.append([attrs["income"], attrs["loan_amount"], attrs["age"]])
        y_list.append(attrs["is_fraud"])

    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)

    # Extract Edge Indices -> Shape (2, E)
    edges = list(G.edges())
    if edges:
        edge_index_list = []
        for u, v in edges:
            edge_index_list.append([u, v])
            edge_index_list.append([v, u])  # Undirected graph
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    if HAS_PYG:
        data = PyGData(x=x, edge_index=edge_index, y=y)
        data.num_nodes = len(nodes_sorted)
        return data
    else:
        # Fallback Python container matching PyG attributes
        class CustomGraphData:
            def __init__(self, x, edge_index, y):
                self.x = x
                self.edge_index = edge_index
                self.y = y
                self.num_nodes = x.size(0)

        return CustomGraphData(x=x, edge_index=edge_index, y=y)


def print_graph_stats(G: nx.Graph):
    """
    Prints descriptive graph statistics.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_components = nx.number_connected_components(G)
    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees)) if degrees else 0.0

    print("\n" + "=" * 45)
    print("        APPLICANT GRAPH STATISTICS          ")
    print("=" * 45)
    print(f"Total Nodes (Applicants)    : {num_nodes}")
    print(f"Total Edges (Shared Attrs)  : {num_edges}")
    print(f"Connected Components        : {num_components}")
    print(f"Average Node Degree         : {avg_degree:.2f}")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    from src.data_simulation import generate_applicant_dataset

    df = generate_applicant_dataset(num_applicants=500)
    G, _ = build_applicant_graph(df)
    print_graph_stats(G)
    data = convert_to_pyg_data(G)
    print(f"Converted Graph PyTorch Tensor Data: Nodes={data.num_nodes}, Edge Index={data.edge_index.shape}")

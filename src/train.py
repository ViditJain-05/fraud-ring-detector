"""
Training Script for Fraud Ring Detector GCN Model.
Handles node-level split, class-weighted cross-entropy loss, and F1 checkpointing.
"""

import os
import sys
import argparse
from typing import Tuple, Any
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_simulation import generate_applicant_dataset
from src.graph_builder import build_applicant_graph, convert_to_pyg_data, print_graph_stats
from src.model import GCN


def create_node_masks(y: torch.Tensor, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates stratified boolean masks (train_mask, val_mask, test_mask) over graph nodes.
    """
    num_nodes = len(y)
    indices = np.arange(num_nodes)
    labels = y.cpu().numpy()

    train_idx, test_val_idx = train_test_split(indices, train_size=train_ratio, stratify=labels, random_state=seed)

    val_relative_ratio = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(test_val_idx, train_size=val_relative_ratio, stratify=labels[test_val_idx], random_state=seed)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def train_gcn_model(
    data: Any,
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    save_path: str = "models/gcn_model.pt"
) -> Tuple[GCN, float]:
    """
    Executes GCN model training loop.

    Returns:
        Tuple[GCN, float]: (Trained GCN Model, Best Validation F1 Score)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    train_mask, val_mask, test_mask = create_node_masks(y)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    # Class Weights to penalize misclassifying minority fraud class (~5%)
    num_neg = (y[train_mask] == 0).sum().item()
    num_pos = (y[train_mask] == 1).sum().item()
    pos_weight = num_neg / max(num_pos, 1)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)

    model = GCN(in_channels=x.size(1), hidden_channels=32, num_classes=2, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Starting GCN Node Classification Training...")
    print("-" * 65)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val Fraud F1':<15}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation Step
        model.eval()
        with torch.no_grad():
            val_out = model(x, edge_index)
            val_loss = criterion(val_out[val_mask], y[val_mask]).item()
            preds = val_out[val_mask].argmax(dim=-1).cpu().numpy()
            val_labels = y[val_mask].cpu().numpy()
            val_f1 = float(f1_score(val_labels, preds, pos_label=1, zero_division=0))

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:<6d} | {loss.item():<10.4f} | {val_loss:<10.4f} | {val_f1:<15.4f}")

        # Checkpoint Saving based on Validation F1 (NOT accuracy)
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)

    print("-" * 65)
    print(f"SUCCESS: Training complete! Best Validation Fraud F1: {best_val_f1:.4f}")
    print(f"Model saved to '{save_path}'.")
    return model, best_val_f1


def main():
    parser = argparse.ArgumentParser(description="Train Fraud Ring GCN Model")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--data-path", type=str, default="data/applicants.csv", help="Path to applicant CSV")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"INFO: Data file '{args.data_path}' not found. Simulating dataset...")
        data = generate_applicant_dataset(num_applicants=1000)
        from src.data_simulation import save_applicants_to_csv
        save_applicants_to_csv(data, args.data_path)
    
    G, _ = build_applicant_graph(args.data_path)
    print_graph_stats(G)

    data = convert_to_pyg_data(G)
    train_gcn_model(data, epochs=args.epochs, lr=args.lr)



if __name__ == "__main__":
    main()

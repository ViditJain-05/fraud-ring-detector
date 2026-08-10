"""
Graph Convolutional Network (GCN) Architecture for Fraud Ring Detection.

Plain-Language Explanation of GCN "Message Passing":
--------------------------------------------------
In standard tabular machine learning, a model looks at each loan applicant in complete isolation.
In a Graph Convolutional Network (GCN), applicants are nodes connected by shared attributes (edges).

"Message Passing" is the process where each node gathers ("aggregates") information and feature embeddings 
from its immediate graph neighbors. During each GCN layer, an applicant node combines its own income/loan 
features with the features of all applicants connected to it via shared bank accounts, addresses, or phone 
numbers. 

This enables the model to detect coordinated fraud rings: even if an individual fraud applicant has a 
"normal-looking" income and loan amount, aggregating messages from connected suspicious nodes alerts 
the network to flag the entire cluster as a fraud ring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class NativeGCNLayer(nn.Module):
    """
    Native PyTorch GCN Message Passing Layer (Fallback when torch_geometric is not installed).
    Computes: Z = D^(-1/2) * (A + I) * D^(-1/2) * X * W
    """

    def __init__(self, in_features: int, out_features: int):
        super(NativeGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        h = self.linear(x)

        if edge_index.numel() == 0:
            return h

        # Build Sparse Normalized Adjacency Matrix with Self-Loops
        row, col = edge_index[0], edge_index[1]
        
        # Add Self-Loops
        self_loop = torch.arange(num_nodes, device=x.device)
        row = torch.cat([row, self_loop])
        col = torch.cat([col, self_loop])

        # Compute Node Degrees
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)

        # Edge Weights (Symmetric Normalization)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Aggregate Neighbor Messages
        out = torch.zeros_like(h)
        out.index_add_(0, row, norm.unsqueeze(-1) * h[col])
        return out


class GCN(nn.Module):
    """
    2-Layer Graph Convolutional Network for Node Binary Classification (Fraud / Legitimate).
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 16, num_classes: int = 2, dropout: float = 0.2):
        super(GCN, self).__init__()
        self.dropout = dropout

        if HAS_PYG:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, num_classes)
        else:
            self.conv1 = NativeGCNLayer(in_channels, hidden_channels)
            self.conv2 = NativeGCNLayer(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GCN.

        Args:
            x (torch.Tensor): Node feature matrix (N, in_channels).
            edge_index (torch.Tensor): Graph edge connectivity index (2, E).

        Returns:
            torch.Tensor: Node classification logits (N, num_classes).
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

"""
Evaluation Script for Fraud Ring Detector GCN Model.

Why Accuracy Alone is Misleading:
---------------------------------
In severely imbalanced fraud detection scenarios (where only ~5% of applicants are fraudulent),
a trivial baseline model that predicts "NOT FRAUD" for EVERY applicant achieves 95.0% accuracy.
However, such a model catches 0 fraud rings, resulting in total failure for risk management.

Therefore, we evaluate model performance using Precision, Recall, F1-Score, and ROC-AUC specifically
focused on the minority Fraud Class (Label=1).
"""

import os
import sys
import argparse
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import torch
import torch.nn.functional as F

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_simulation import generate_applicant_dataset
from src.graph_builder import build_applicant_graph, convert_to_pyg_data
from src.model import GCN
from src.train import create_node_masks


def evaluate_gcn(
    model_path: str = "models/gcn_model.pt",
    data_path: str = "data/applicants.csv",
    save_dir: str = "assets"
):
    """
    Evaluates trained GCN model on test nodes and computes classification metrics.
    """
    if not os.path.exists(data_path):
        data = generate_applicant_dataset(num_applicants=1000)
        from src.data_simulation import save_applicants_to_csv
        save_applicants_to_csv(data, data_path)

    G, _ = build_applicant_graph(data_path)
    data = convert_to_pyg_data(G)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    _, _, test_mask = create_node_masks(y)
    test_mask = test_mask.to(device)

    model = GCN(in_channels=x.size(1), hidden_channels=32, num_classes=2).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from '{model_path}'.")
    else:
        print(f"WARNING: Model file '{model_path}' not found. Using initialized network.")

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = F.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

    y_test = y[test_mask].cpu().numpy()
    preds_test = preds[test_mask].cpu().numpy()
    probs_test = probs[test_mask].cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds_test, pos_label=1, average="binary", zero_division=0)
    
    try:
        auc_score = float(roc_auc_score(y_test, probs_test))
    except Exception:
        auc_score = 0.5

    cm = confusion_matrix(y_test, preds_test)

    print("\n" + "=" * 55)
    print("      GCN FRAUD CLASS EVALUATION REPORT      ")
    print("=" * 55)
    print(f"Test Set Size              : {len(y_test)} nodes")
    print(f"Actual Fraud Cases         : {np.sum(y_test == 1)}")
    print(f"Fraud Precision            : {precision * 100:.2f}%")
    print(f"Fraud Recall               : {recall * 100:.2f}%")
    print(f"Fraud F1-Score             : {f1 * 100:.2f}%")
    print(f"Fraud ROC-AUC              : {auc_score:.4f}")
    print("=" * 55)

    print("\nFull Sklearn Classification Report:\n")
    print(classification_report(y_test, preds_test, target_names=["Legitimate (0)", "Fraud Ring (1)"]))

    # Save Confusion Matrix Plot
    if HAS_MATPLOTLIB:
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
        fig.colorbar(cax)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

        ax.set_xticklabels([''] + ["Legitimate", "Fraud"])
        ax.set_yticklabels([''] + ["Legitimate", "Fraud"])
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('Actual True Label', fontsize=12)
        plt.title('Fraud Ring Detection Confusion Matrix', fontsize=13)
        plt.tight_layout()

        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix plot to '{cm_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fraud Ring GCN Model")
    parser.add_argument("--model-path", type=str, default="models/gcn_model.pt", help="Path to trained GCN model")
    parser.add_argument("--data-path", type=str, default="data/applicants.csv", help="Path to applicant CSV")
    args = parser.parse_args()

    evaluate_gcn(model_path=args.model_path, data_path=args.data_path)


if __name__ == "__main__":
    main()

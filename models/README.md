# Models Directory

This directory stores trained Graph Convolutional Network (GCN) model weights for the **Fraud Ring Detector**.

- `gcn_model.pt`: Saved PyTorch state_dict for node classification.
- Model Architecture: 2-Layer GCN (`GCNConv` / Message Passing) trained with class-weighted cross-entropy loss.

To retrain the GCN model, execute:
```bash
python src/train.py --epochs 80 --lr 0.01
```

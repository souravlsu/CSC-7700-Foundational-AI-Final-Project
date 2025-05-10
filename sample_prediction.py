import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from model_ResFNN import ResNetFeedForwardNN as FF

# ─── Settings ────────────────────────────────────────────────────────────────
# Automatically selecting GPU if available; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─── Re‑defining build_dataloaders to get test set ──────────────────────────────
def build_dataloaders(X: np.ndarray, Y: np.ndarray, batch_size: int = 5000):
    # Combining features and targets for consistent shuffling/splitting
    data = np.hstack([X, Y])
    n = len(data)

    # Splitting into 70% train, 15% val, 15% test
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    train, val, test = np.split(data, [n_train, n_train + n_val])

    # Fitting MinMaxScaler on training data only, then transform all sets
    scaler = MinMaxScaler().fit(train)
    train_s = scaler.transform(train)
    val_s   = scaler.transform(val)
    test_s  = scaler.transform(test)

    # Converting to PyTorch tensors and move to device
    T_train = torch.tensor(train_s, dtype=torch.float32).to(device)
    T_val   = torch.tensor(val_s,   dtype=torch.float32).to(device)
    T_test  = torch.tensor(test_s,  dtype=torch.float32).to(device)

    # Returning data loaders and the fitted scaler
    return (
        DataLoader(T_train, batch_size=batch_size, shuffle=True),
        DataLoader(T_val,   batch_size=len(val),     shuffle=False),
        DataLoader(T_test,  batch_size=len(test),    shuffle=False),
        scaler
    )

if __name__ == "__main__":
    # ─── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv("model_data.csv")
    orig_feats  = df.columns[:4].tolist()        # Input feature columns
    all_targets = df.columns[4:].tolist()        # All target columns

    # Splitting targets into regular and special groups (same as training pipeline)
    special = [all_targets[18], all_targets[23]]
    regular = [col for col in all_targets if col not in special]

    # Extract features and corresponding regular targets
    X_reg = df[orig_feats].values
    Y_reg = df[regular].values

    # ─── Building test loader ─────────────────────────────────────────────────────
    _, _, test_loader, scaler = build_dataloaders(X_reg, Y_reg)

    # ─── Loading saved model ──────────────────────────────────────────────────────
    # Initializing the model architecture and load pretrained weights
    model_reg = FF(
        input_dim=len(orig_feats),
        output_dim=len(regular),
        hidden_dim=[512, 512, 1024, 512],
        criterion="MSE",
        activation_type="leakyrelu",
        regularization="dropout"
    ).to(device)
    model_reg.load_state_dict(torch.load("saved_models/model_reg.pth", map_location=device))
    model_reg.eval()  # Set model to evaluation mode

    # ─── Predicting on the test set ───────────────────────────────────────────────
    actual_tensor, pred_tensor = model_reg.test_model(model_reg, test_loader)
    y_true = actual_tensor.detach().cpu().numpy()
    y_pred = pred_tensor.detach().cpu().numpy()

    # ─── Preparing sample rows ───────────────────────────────────────────────────
    # Comparing actual vs. predicted for the first test instance
    actual_first = y_true[0]
    pred_first   = y_pred[0]

    # Creating a DataFrame showing Actual vs Predicted values
    sample_df = pd.DataFrame(
        np.vstack([actual_first, pred_first]),
        index=["Actual", "Predicted"],
        columns=regular
    )

    # ─── Display ───────────────────────────────────────────────────────────────
    print("Sample prediction (first test instance):")
    print(sample_df)

    # Saving the sample comparison to CSV
    outpath = os.path.join("plots", "sample_prediction.csv")
    os.makedirs("plots", exist_ok=True)
    sample_df.to_csv(outpath)
    print(f"Saved sample prediction to {outpath}")

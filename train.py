import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from model_ResFNN import ResNetFeedForwardNN as FF
from utils import myutils

warnings.simplefilter(action='ignore', category=FutureWarning)

# ─── Selecting device ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ─── Building DataLoaders ──────────────────────────────────────────────────────────
def build_dataloaders(X: np.ndarray, Y: np.ndarray, batch_size: int = 5000):
    # Creating and splitting dataset
    data = np.hstack([X, Y])
    n = len(data)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    train, val, test = np.split(data, [n_train, n_train + n_val])

    # Scaling data using MinMaxScaler
    scaler = MinMaxScaler().fit(train)
    train_s = scaler.transform(train)
    val_s   = scaler.transform(val)
    test_s  = scaler.transform(test)

    # Converting to PyTorch tensors
    T_train = torch.tensor(train_s, dtype=torch.float32).to(device)
    T_val   = torch.tensor(val_s,   dtype=torch.float32).to(device)
    T_test  = torch.tensor(test_s,  dtype=torch.float32).to(device)

    # Creating DataLoaders
    return (
        DataLoader(T_train, batch_size=batch_size, shuffle=True),
        DataLoader(T_val,   batch_size=len(val),   shuffle=False),
        DataLoader(T_test,  batch_size=len(test),  shuffle=False),
        scaler
    )

# ─── Training model (returning per-epoch losses) ───────────────────────────────────
def train_model(model, loaders, optimizer, num_epochs=10, patience=5, save_path=None):
    train_loader, val_loader, *_ = loaders
    best_val = float('inf')
    wait = 0

    train_losses = []
    val_losses   = []

    for epoch in range(num_epochs):
        _, train_loss = model.train_model(model, train_loader, optimizer)  # Training
        val_loss      = model.eval_model(model, val_loader)                # Validating

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train MSE: {train_loss:.6f}  Val MSE: {val_loss:.6f}")

        # Saving best model and checking early stopping
        if val_loss < best_val:
            best_val, wait = val_loss, 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return train_losses, val_losses

# ─── Plotting epoch losses ─────────────────────────────────────────────────────────
def plot_epoch_losses(train_losses, val_losses, prefix):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train MSE', color='blue')
    plt.plot(val_losses,   label='Val MSE', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'{prefix} — Training vs Validation MSE')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'plots/{prefix}_epoch_losses.png', dpi=300)
    plt.close()
    print(f"\n Saving epoch-loss plot: plots/{prefix}_epoch_losses.png")

# ─── Generating parity plots for testing ───────────────────────────────────────────
def test_and_plot(model, loaders, target_names, prefix):
    _, _, test_loader, _ = loaders
    model.load_state_dict(torch.load(f"saved_models/{prefix}.pth", map_location=device))
    model.to(device)
    model.eval()

    # Running predictions
    actual, pred = model.test_model(model, test_loader)
    y_true = actual.detach().cpu().numpy()
    y_pred = pred.detach().cpu().numpy()
    helpers = myutils(target_names)

    sns.set(style="white")
    for i, tname in enumerate(target_names):
        dfp = pd.DataFrame({'Actual': y_true[:, i], 'Predicted': y_pred[:, i]})
        mn, mx = dfp.values.min(), dfp.values.max()
        line = np.linspace(mn, mx, 100)

        plt.figure(figsize=(5, 5))
        sns.scatterplot(data=dfp, x='Actual', y='Predicted', s=15)
        plt.plot(line, line, 'k--', linewidth=1)
        mse = helpers.mean_squared_error(y_true[:, i], y_pred[:, i])
        r2  = helpers.r_squared(y_true[:, i], y_pred[:, i])
        plt.title(f"Target: {tname}  MSE: {mse:.5f}  R²: {r2:.3f}")
        plt.tight_layout()
        plt.savefig(f"plots/{tname}.png", dpi=300)
        plt.close()

# ─── Running main pipeline ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Loading dataset and preparing features
    df = pd.read_csv("model_data.csv")
    orig_feats  = df.columns[:4].tolist()
    all_targets = df.columns[4:].tolist()
    special = [all_targets[18], all_targets[23]]
    regular = [col for col in all_targets if col not in special]

    # Creating pipeline for regular targets
    X_reg = df[orig_feats].values
    Y_reg = df[regular].values
    loaders_reg = build_dataloaders(X_reg, Y_reg)

    # Creating pipeline for feature-engineered targets
    df_eng = df.copy()
    df_eng["Zmean_x_C"]   = df_eng["Zmean"] * df_eng["C"]
    df_eng["Zmean_div_C"] = df_eng["Zmean"] / (df_eng["C"] + 1e-6)
    df_eng["C_squared"]   = df_eng["C"] ** 2
    eng_feats = orig_feats + ["Zmean_x_C", "Zmean_div_C", "C_squared"]

    X_sp = df_eng[eng_feats].values
    Y_sp = df_eng[special].values
    loaders_sp = build_dataloaders(X_sp, Y_sp)

    # Setting training hyperparameters
    hidden_dims = [512, 512, 1024, 512]
    activation  = "leakyrelu"
    reg_type    = "dropout"
    criterion   = "MSE"
    optim_fn    = lambda params: torch.optim.Adam(params, lr=1e-3)

    # Training and evaluating on regular targets
    print("\n=== TRAINING ON REGULAR TARGETS ===")
    model_reg = FF(
        input_dim=len(orig_feats),
        output_dim=len(regular),
        hidden_dim=hidden_dims,
        criterion=criterion,
        activation_type=activation,
        regularization=reg_type
    ).to(device)
    opt_reg = optim_fn(model_reg.parameters())
    train_losses_reg, val_losses_reg = train_model(
        model_reg, loaders_reg, opt_reg,
        num_epochs=100, patience=10,
        save_path="saved_models/model_reg.pth"
    )
    plot_epoch_losses(train_losses_reg, val_losses_reg, prefix="Regular Targets")
    test_and_plot(model_reg, loaders_reg, regular, prefix="model_reg")

    # Training and evaluating on feature-engineered targets
    print("\n=== TRAINING ON FEATURE-ENGINEERED TARGETS ===")
    model_sp = FF(
        input_dim=len(eng_feats),
        output_dim=len(special),
        hidden_dim=hidden_dims,
        criterion=criterion,
        activation_type=activation,
        regularization=reg_type
    ).to(device)
    opt_sp = optim_fn(model_sp.parameters())
    train_losses_sp, val_losses_sp = train_model(
        model_sp, loaders_sp, opt_sp,
        num_epochs=100, patience=10,
        save_path="saved_models/model_sp.pth"
    )
    plot_epoch_losses(train_losses_sp, val_losses_sp, prefix="Feature Engineered Targets")
    test_and_plot(model_sp, loaders_sp, special, prefix="model_sp")

    print("\n Finished. Checking `plots/` for all loss curves and parity plots.")

# Flamelet Surrogate Model using Residual Feedforward Neural Network

This repository provides a complete pipeline for training, validating, and evaluating a deep residual feedforward neural network to approximate the output of a flamelet-generated manifold. It includes feature engineering for difficult targets and demonstrates model performance with parity plots and sample predictions.

## Overview

The goal of this project is to replace large multidimensional interpolation tables used in combustion simulations with a fast and accurate machine learning surrogate model. A residual feedforward neural network is trained to predict 25 target variables. Special handling is applied to two poorly performing targets using additional engineered features.

## Repository Structure

```
├── train.py                 # Full training pipeline including feature engineering and evaluation
├── sample_prediction.py    # Script to test saved model on unseen data and show sample prediction
├── model_ResFNN.py         # Residual feedforward neural network architecture
├── utils.py                # Visualization and metric helper utilities
├── model_data.csv          # Required dataset (not included in repo)
├── saved_models/           # Directory to store trained model checkpoints
├── plots/                  # Directory for saving plots and sample outputs
```

## Features

- Residual connections to improve training stability
- Customizable architecture: activation, dropout, and optimizer
- Feature engineering for specific targets using domain knowledge
- Training/validation/test split with scaling
- Parity plots and loss curves to assess model performance
- Prediction interface to inspect model behavior on new samples

## Requirements

- Python 3.8+
- PyTorch
- pandas
- scikit-learn
- matplotlib
- seaborn

Install required libraries via:

```bash
pip install torch pandas scikit-learn matplotlib seaborn
```

## Usage

### 1. Train Models

Run the full training pipeline and generate results:

```bash
python train.py
```

This will:
- Train a model for regular targets
- Train a separate model for two difficult targets using feature engineering
- Save the models in `saved_models/`
- Save training/validation loss curves and parity plots in `plots/`

### 2. Run Sample Prediction

After training, you can inspect predictions from the test set:

```bash
python sample_prediction.py
```

It will display a DataFrame of actual vs. predicted values for one test instance and save it to `plots/sample_prediction.csv`.

## Customization

You can modify:
- Network structure: adjust `hidden_dims` in `train.py`
- Activation function: `'relu'`, `'leakyrelu'`, `'tanh'`, `'sigmoid'`
- Regularization method: `'dropout'` or `None`
- Learning rate and optimizer

## Citation

If you use this codebase or the methodology in your work, please consider citing or acknowledging the repository.

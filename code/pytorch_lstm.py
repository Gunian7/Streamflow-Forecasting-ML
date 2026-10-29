"""PyTorch LSTM regression utilities for time-series forecasting."""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """Tensor dataset accepting either (samples, features) or windowed input."""

    def __init__(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, None, :]
        if X.ndim != 3:
            raise ValueError("X must have shape (samples, features) or (samples, steps, features)")
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or len(X) != len(y):
            raise ValueError("X and y must have the same sample count; y must be 1D or 2D")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values")
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    """LSTM regressor supporting stacked/bidirectional networks and multi-output."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1,
                 dropout=0.2, bidirectional=False):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers, output_dim) < 1:
            raise ValueError("input_dim, hidden_dim, num_layers and output_dim must be positive")
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * directions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, features) or (batch, steps, features)")
        sequence, _ = self.lstm(x)
        output = self.head(sequence[:, -1, :])
        return output.squeeze(-1) if self.output_dim == 1 else output


class LSTMTrainer:
    """Fit and predict with train-only scaling, early stopping and LR scheduling."""

    def __init__(self, model, device="cpu", lr=1e-3, scheduler_patience=5):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=scheduler_patience
        )
        self.criterion = nn.MSELoss()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = []

    def _scale_X(self, X, fit=False):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, None, :]
        shape = X.shape
        flat = X.reshape(-1, shape[-1])
        scaled = self.scaler_X.fit_transform(flat) if fit else self.scaler_X.transform(flat)
        return scaled.reshape(shape).astype(np.float32)

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32,
            early_stopping_patience=15, verbose=False):
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must either both be provided or both omitted")
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        X_scaled = self._scale_X(X, fit=True)
        y_scaled = self.scaler_y.fit_transform(y).astype(np.float32)
        loader = DataLoader(TimeSeriesDataset(X_scaled, y_scaled), batch_size=batch_size,
                            shuffle=True)
        val_loader = None
        if X_val is not None:
            y_val = np.asarray(y_val, dtype=np.float32)
            if y_val.ndim == 1:
                y_val = y_val[:, None]
            val_loader = DataLoader(
                TimeSeriesDataset(self._scale_X(X_val), self.scaler_y.transform(y_val)),
                batch_size=batch_size, shuffle=False,
            )

        best_loss, best_state, stale_epochs = float("inf"), None, 0
        self.history = []
        for epoch in range(epochs):
            self.model.train()
            train_total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                prediction = self.model(xb)
                if prediction.ndim == 1:
                    prediction = prediction[:, None]
                loss = self.criterion(prediction, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_total += loss.item() * len(xb)
            train_loss = train_total / len(loader.dataset)

            val_loss = train_loss
            if val_loader is not None:
                self.model.eval()
                val_total = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        prediction = self.model(xb)
                        if prediction.ndim == 1:
                            prediction = prediction[:, None]
                        val_total += self.criterion(prediction, yb).item() * len(xb)
                val_loss = val_total / len(val_loader.dataset)
                self.scheduler.step(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            self.history.append({"epoch": epoch + 1, "train_loss": train_loss,
                                 "val_loss": val_loss,
                                 "lr": self.optimizer.param_groups[0]["lr"]})
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Epoch {epoch + 1}/{epochs}: train={train_loss:.5f}, val={val_loss:.5f}")
            if val_loader is not None and stale_epochs >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        X_scaled = self._scale_X(X)
        loader = DataLoader(torch.from_numpy(X_scaled), batch_size=256, shuffle=False)
        outputs = []
        with torch.no_grad():
            for xb in loader:
                outputs.append(self.model(xb.to(self.device)).cpu().numpy())
        pred_scaled = np.concatenate(outputs, axis=0)
        if pred_scaled.ndim == 1:
            pred_scaled = pred_scaled[:, None]
        pred = self.scaler_y.inverse_transform(pred_scaled)
        return pred.ravel() if pred.shape[1] == 1 else pred


if __name__ == "__main__":
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5, 4)).astype(np.float32)
    y = X[:, -1, 0] * 2 - X[:, -2, 1]
    model = LSTMRegressor(input_dim=4, hidden_dim=16, num_layers=2,
                          output_dim=1, dropout=0.2, bidirectional=True)
    trainer = LSTMTrainer(model)
    trainer.fit(X[:64], y[:64], X[64:], y[64:], epochs=8, verbose=False)
    assert trainer.predict(X[64:]).shape == (16,)
    print("PyTorch LSTM smoke test passed")

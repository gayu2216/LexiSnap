"""
lstm_predictor.py
=================
Drop-in inference module for the trained SentimentDirectionLSTM.

Usage in app.py:
    from lstm_predictor import load_lstm, predict_direction

    _lstm = load_lstm("lstm_run")   # folder with best_model.pt, scaler.pkl, config.json
    result = predict_direction(_lstm, sentiment_score=0.78, category_id=0, market_row=df_row)
    # result = {"SPY": "Up", "QQQ": "Up", "TLT": "Neutral", "GLD": "Down",
    #           "SPY_probs": [0.12, 0.21, 0.67], ...}
"""

from __future__ import annotations
import json, pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ── Model definition (must match training exactly) ────────────────────────────

HEAD_TICKERS   = ["SPY", "QQQ", "TLT", "GLD"]
ALL_TICKERS    = ["SPY", "QQQ", "TLT", "GLD", "USO", "UVXY"]
FEAT_SUFFIXES  = ["price", "rsi14", "macd", "bb", "volratio"]
DIR_LABELS     = {0: "Down", 1: "Neutral", 2: "Up"}
DIR_COLORS     = {"Up": "#0f9d78", "Neutral": "#5c7089", "Down": "#cc4b4b"}
N_CLASSES      = 3
N_CATS         = 5


class _DirectionHead(nn.Module):
    def __init__(self, trunk_dim: int, n_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )
    def forward(self, x): return self.net(x)


class _SentimentDirectionLSTM(nn.Module):
    def __init__(self, n_features: int, n_cats: int = N_CATS, embed_dim: int = 8,
                 hidden: int = 64, n_layers: int = 2, trunk_dim: int = 32,
                 dropout: float = 0.5):
        super().__init__()
        self.embed = nn.Embedding(n_cats, embed_dim)
        self.lstm  = nn.LSTM(n_features + embed_dim, hidden, n_layers,
                             batch_first=True,
                             dropout=dropout if n_layers > 1 else 0.0)
        self.trunk = nn.Sequential(
            nn.Linear(hidden, trunk_dim), nn.ReLU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList([
            _DirectionHead(trunk_dim, N_CLASSES, dropout) for _ in HEAD_TICKERS
        ])

    def forward(self, feat, cat):
        emb  = self.embed(cat)
        x    = torch.cat([feat, emb], dim=-1)
        x, _ = self.lstm(x.unsqueeze(1))
        x    = self.trunk(x.squeeze(1))
        return torch.stack([h(x) for h in self.heads], dim=1)   # (B, 4, 3)


# ── Public API ────────────────────────────────────────────────────────────────

def load_lstm(run_dir: str) -> Optional[dict]:
    """
    Load model, scaler and config from a run directory.
    Returns a bundle dict, or None on failure.

    run_dir should contain:
        best_model.pt
        scaler.pkl
        config.json
    """
    run_path = Path(run_dir)
    try:
        cfg_path    = run_path / "config.json"
        scaler_path = run_path / "scaler.pkl"
        model_path  = run_path / "best_model.pt"

        missing = [p.name for p in [cfg_path, scaler_path, model_path] if not p.exists()]
        if missing:
            return {"error": f"Missing files in {run_dir}: {missing}"}

        cfg = json.loads(cfg_path.read_text())
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        device    = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt      = torch.load(model_path, map_location=device)
        n_feat    = cfg.get("n_features") or ckpt.get("n_features", 31)
        model     = _SentimentDirectionLSTM(
            n_features = n_feat,
            n_cats     = cfg.get("n_cats",     N_CATS),
            embed_dim  = cfg.get("embed_dim",  8),
            hidden     = cfg.get("hidden",     64),
            n_layers   = cfg.get("n_layers",   2),
            trunk_dim  = cfg.get("trunk_dim",  32),
            dropout    = cfg.get("dropout",    0.5),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        model.to(device)

        return {
            "model":      model,
            "scaler":     scaler,
            "cfg":        cfg,
            "feat_cols":  cfg.get("feat_cols", []),
            "device":     device,
            "error":      None,
        }
    except Exception as e:
        return {"error": str(e)}


def predict_direction(
    bundle: dict,
    sentiment_score: float,
    category_id: int,
    market_snapshot: Optional[dict] = None,
) -> dict:
    """
    Run one prediction.

    Parameters
    ----------
    bundle          : returned by load_lstm()
    sentiment_score : 0.0 – 1.0  (from DomainSentimentPipeline)
    category_id     : 0–4  (Business=0, Tech=1, Politics=2, Law=3, Other=4)
    market_snapshot : dict of {col_name: value} for technical features.
                      If None or partial, missing values are filled with 0.

    Returns
    -------
    {
      "SPY": "Up",  "SPY_probs": {"Down":0.12,"Neutral":0.21,"Up":0.67},
      "QQQ": "Neutral", ...
      "error": None
    }
    """
    if bundle is None or bundle.get("error"):
        return {"error": bundle.get("error", "Model not loaded") if bundle else "Model not loaded"}

    model     = bundle["model"]
    scaler    = bundle["scaler"]
    feat_cols = bundle["feat_cols"]
    device    = bundle["device"]

    # Build feature vector in same order as training
    snap = market_snapshot or {}
    row  = {"sentiment_score": sentiment_score}
    for t in ALL_TICKERS:
        for s in FEAT_SUFFIXES:
            col = f"{t}_{s}"
            row[col] = snap.get(col, np.nan)

    # Align to feat_cols used during training (fills any gaps with 0)
    vec = np.array([row.get(c, np.nan) for c in feat_cols], dtype=np.float32)
    # Replace NaN with 0 (same as median-fill used in training on unseen data)
    vec = np.where(np.isnan(vec), 0.0, vec)
    vec_scaled = scaler.transform(vec.reshape(1, -1)).astype(np.float32)

    feat_t = torch.tensor(vec_scaled, dtype=torch.float32).to(device)
    cat_t  = torch.tensor([category_id], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(feat_t, cat_t)           # (1, 4, 3)
        probs  = torch.softmax(logits, dim=-1)  # (1, 4, 3)

    result = {"error": None}
    for i, ticker in enumerate(HEAD_TICKERS):
        p    = probs[0, i].cpu().numpy()
        pred = int(p.argmax())
        result[ticker]             = DIR_LABELS[pred]
        result[f"{ticker}_probs"]  = {
            "Down":    round(float(p[0]), 3),
            "Neutral": round(float(p[1]), 3),
            "Up":      round(float(p[2]), 3),
        }
        result[f"{ticker}_confidence"] = round(float(p.max()), 3)

    return result
# Neural ODEs for Time-Series Forecasting

This project implements Neural Ordinary Differential Equations (Neural ODEs) to model and forecast time-series data. Neural ODEs can be applied to finance, physics, and electrical engineering data.

## Project Structure

- `generate_dataset.py` – Generates synthetic time-series datasets (`data/sample_large.csv`).  
- `view_model.py` – Basic script to view predictions on `sample_large.csv`.  
- `view_model_yfinance.py` – Downloads stock data from Yahoo Finance and visualizes predictions.  
- `view_model_combined.py` – Predicts and plots results using either Yahoo Finance or local datasets.  
- `src/train.py` – Trains Neural ODE on a dataset and saves model to `results/`.  
- `src/data.py` – Loads CSV, normalizes data, and splits into sequences for training/testing.  
- `src/eval.py` – Evaluates model on test data.  
- `src/models/neural_ode.py` – Implements Neural ODE architecture.  
- `src/models/odefunc.py` – Defines the ODE function used in Neural ODE.  
- `src/models/latent_ode.py` – Latent Neural ODE (advanced architecture).  
- `src/models/cde_model.py` – Neural Controlled Differential Equation model.  
- `src/models/lstm.py` – LSTM baseline model.  
- `data/` – Folder containing CSV datasets (e.g., `sample_large.csv`, `aapl.csv`).  
- `results/` – Saves trained models (`.pt`) and prediction plots (`.png`).  
- `LICENSE` – MIT License.  
- `requirements.txt` – Python dependencies.  

## Setup

1. Activate virtual environment:
```bash
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run scripts in src/ to preprocess data, train models, and evaluate results.

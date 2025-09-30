# src/data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path, feature_col='value', window_size=20, test_size=0.2):
    """
    Load and preprocess time-series data for Neural ODE.

    Args:
        file_path (str): Path to CSV file
        feature_col (str): Name of the column with time-series values
        window_size (int): Number of past steps used for prediction
        test_size (float): Fraction of data used for testing

    Returns:
        X_train, y_train, X_test, y_test : np.arrays
    """
    # Load CSV
    df = pd.read_csv(file_path)
    data = df[feature_col].values.reshape(-1, 1)

    # Normalize data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    
    X = np.array(X)
    y = np.array(y)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    # Example usage
    X_train, y_train, X_test, y_test, scaler = load_data("data/sample.csv")
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

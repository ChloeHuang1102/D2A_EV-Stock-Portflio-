#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM-Robust Portfolio System
Complete Predict-Then-Optimize Framework
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import cvxpy as cp
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully")

# ==========================================
# 1. Historical Data Acquisition
# ==========================================

def get_ev_data(n_stocks=30, start_date='2018-01-01', end_date='2024-01-01'):
    """Get EV stock historical data"""
    print("="*50)
    print("1. Historical Data Acquisition")
    print("="*50)
    
    ev_stocks = [
        'TSLA', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'F', 'GM', 'FORD',
        'RACE', 'BMWYY', 'VWAGY', 'TM', 'HMC', 'BYDDF', 'BAMXF', 'DMLRY',
        'STLA', 'NSANY', 'HYMTF', 'RIVN', 'LCID', 'F', 'GM', 'FORD',
        'TSLA', 'NIO', 'XPEV', 'LI', 'RIVN'
    ]
    
    stock_data = {}
    successful_stocks = []
    
    for symbol in ev_stocks[:n_stocks]:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty and len(data) > 1000:
                stock_data[symbol] = data['Close']
                successful_stocks.append(symbol)
                print(f"✓ {symbol}: {len(data)} trading days")
            else:
                print(f"✗ {symbol}: insufficient data")
        except Exception as e:
            print(f"✗ {symbol}: error - {e}")
    
    raw_data = pd.DataFrame(stock_data).dropna()
    returns = raw_data.pct_change().dropna()
    
    print(f"\nData acquisition completed: {len(successful_stocks)} stocks, {raw_data.shape[0]} trading days")
    return raw_data, returns, successful_stocks

# Get data
raw_data, returns, stock_names = get_ev_data()

# ==========================================
# 2. Data Preprocessing
# ==========================================

print("="*50)
print("2. Data Preprocessing")
print("="*50)

# Standardize price data
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(raw_data)
scaled_prices = pd.DataFrame(scaled_prices, columns=raw_data.columns, index=raw_data.index)

print(f"Price data standardization completed: {scaled_prices.shape}")
print(f"Returns data shape: {returns.shape}")
print("Data preprocessing completed")

# ==========================================
# 3. Data Splitting
# ==========================================

print("="*50)
print("3. Data Splitting")
print("="*50)

# Splitting parameters
test_size = 30
sequence_length = 20

# Split data
train_data = scaled_prices.iloc[:-test_size]
test_data = scaled_prices.iloc[-test_size:]

print(f"Training set: {train_data.shape}")
print(f"Test set: {test_data.shape}")
print(f"Sequence length: {sequence_length}")
print("Data splitting completed")

# ==========================================
# 4. LSTM Model Training
# ==========================================

print("="*50)
print("4. LSTM Model Training")
print("="*50)

def prepare_lstm_data(data, seq_length):
    """Prepare LSTM training data"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data.iloc[i-seq_length:i].values)
        y.append(data.iloc[i].values)
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, n_stocks):
    """Build LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_stocks, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Prepare training data
X_train, y_train = prepare_lstm_data(train_data, sequence_length)
X_val, y_val = prepare_lstm_data(test_data, sequence_length)

# Build and train model
lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]), len(stock_names))

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print("Starting LSTM model training...")

# Train model
history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

print("LSTM model training completed")

# ==========================================
# 5. Rolling Window Prediction
# ==========================================

print("="*50)
print("5. Rolling Window Prediction")
print("="*50)

def rolling_prediction(model, test_data, seq_length):
    """Rolling window prediction"""
    predictions = []
    
    for i in range(seq_length, len(test_data)):
        # Get window data
        window_data = test_data.iloc[i-seq_length:i]
        X_input = window_data.values.reshape(1, seq_length, -1)
        
        # Predict
        pred = model.predict(X_input, verbose=0)
        predictions.append(pred[0])
    
    return np.array(predictions)

# Execute rolling prediction
predictions = rolling_prediction(lstm_model, test_data, sequence_length)

print(f"Prediction results shape: {predictions.shape}")
print("Rolling window prediction completed")

# ==========================================
# 6. Uncertainty Quantification
# ==========================================

print("="*50)
print("6. Uncertainty Quantification")
print("="*50)

def quantify_uncertainty(model, test_data, seq_length, n_samples=10):
    """Quantify prediction uncertainty"""
    predictions_list = []
    
    for i in range(seq_length, len(test_data)):
        window_data = test_data.iloc[i-seq_length:i]
        X_input = window_data.values.reshape(1, seq_length, -1)
        
        # Multiple predictions
        pred_samples = []
        for _ in range(n_samples):
            pred = model.predict(X_input, verbose=0)
            pred_samples.append(pred[0])
        
        pred_samples = np.array(pred_samples)
        mean_pred = np.mean(pred_samples, axis=0)
        uncertainty = np.std(pred_samples, axis=0)
        
        predictions_list.append((mean_pred, uncertainty))
    
    return predictions_list

# Quantify uncertainty
uncertainty_results = quantify_uncertainty(lstm_model, test_data, sequence_length)

mean_predictions = np.array([result[0] for result in uncertainty_results])
uncertainties = np.array([result[1] for result in uncertainty_results])

print(f"Mean predictions shape: {mean_predictions.shape}")
print(f"Uncertainties shape: {uncertainties.shape}")
print("Uncertainty quantification completed")

# ==========================================
# 7. Robust Optimization
# ==========================================

print("="*50)
print("7. Robust Optimization")
print("="*50)

def robust_optimization(expected_returns, uncertainty, cov_matrix, risk_aversion=1.0, uncertainty_factor=0.1):
    """Robust Mean-Variance optimization"""
    n = len(expected_returns)
    weights = cp.Variable(n)
    
    # Robust returns considering uncertainty
    robust_returns = expected_returns - uncertainty_factor * uncertainty
    
    # Objective function
    portfolio_return = cp.sum(cp.multiply(weights, robust_returns))
    portfolio_risk = cp.quad_form(weights, cov_matrix)
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
    
    # Constraints
    constraints = [
        cp.sum(weights) == 1,  # Weights sum to 1
        weights >= 0,  # No short selling
        weights <= 0.1,  # Maximum 10% per stock
    ]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if problem.status == cp.OPTIMAL:
        return weights.value
    else:
        return np.ones(n) / n  # Equal weights

print("Robust optimization function defined")

# ==========================================
# 8. Portfolio Weights
# ==========================================

print("="*50)
print("8. Portfolio Weights")
print("="*50)

# Calculate covariance matrix
cov_matrix = returns.cov().values

# Calculate optimal weights for each prediction time point
portfolio_weights = []

for i in range(len(mean_predictions)):
    expected_returns = mean_predictions[i]
    uncertainty = uncertainties[i]
    
    # Robust optimization
    optimal_weights = robust_optimization(expected_returns, uncertainty, cov_matrix)
    portfolio_weights.append(optimal_weights)

portfolio_weights = np.array(portfolio_weights)

print(f"Portfolio weights shape: {portfolio_weights.shape}")
print(f"Weight sums: {portfolio_weights.sum(axis=1)[:5]}...")
print("Portfolio weights calculation completed")

# ==========================================
# 9. Backtesting Evaluation
# ==========================================

print("="*50)
print("9. Backtesting Evaluation")
print("="*50)

# Calculate portfolio returns
portfolio_returns = []
actual_returns = returns.iloc[-len(portfolio_weights):].values

for i, weights in enumerate(portfolio_weights):
    if i < len(actual_returns):
        portfolio_return = np.dot(weights, actual_returns[i])
        portfolio_returns.append(portfolio_return)

portfolio_returns = np.array(portfolio_returns)

# Calculate performance metrics
total_return = (1 + portfolio_returns).prod() - 1
volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

# Maximum drawdown
cumulative = (1 + portfolio_returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()

print(f"Portfolio returns shape: {portfolio_returns.shape}")
print("Backtesting evaluation completed")

# ==========================================
# 10. Performance Report
# ==========================================

print("="*60)
print("10. Performance Report")
print("="*60)

print(f"Investment targets: {len(stock_names)} stocks")
print(f"Test period: {len(portfolio_returns)} trading days")
print(f"Sequence length: {sequence_length} days")

print("\n" + "-"*60)
print("Performance Metrics")
print("-"*60)
print(f"• Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
print(f"• Annualized Volatility: {volatility:.4f} ({volatility*100:.2f}%)")
print(f"• Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"• Maximum Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")

print("\n" + "-"*60)
print("Model Features")
print("-"*60)
print("• LSTM Prediction: Captures time series patterns")
print("• Uncertainty Quantification: Considers prediction risk")
print("• Robust Optimization: Maximizes robust returns")
print("• Risk Control: Limits concentration risk")

# Visualization
plt.figure(figsize=(15, 10))

# Portfolio value curve
plt.subplot(2, 2, 1)
portfolio_values = (1 + portfolio_returns).cumprod()
plt.plot(portfolio_values, label='Portfolio Value', linewidth=2)
plt.title('Portfolio Value Curve')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Return distribution
plt.subplot(2, 2, 2)
plt.hist(portfolio_returns, bins=20, alpha=0.7, edgecolor='black')
plt.title('Portfolio Return Distribution')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)

# Weight distribution
plt.subplot(2, 2, 3)
avg_weights = portfolio_weights.mean(axis=0)
plt.bar(range(len(stock_names)), avg_weights)
plt.title('Average Portfolio Weights')
plt.xlabel('Stocks')
plt.ylabel('Weight')
plt.xticks(range(len(stock_names)), stock_names, rotation=45)
plt.grid(True)

# Drawdown curve
plt.subplot(2, 2, 4)
plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
plt.plot(drawdown, color='red', linewidth=1)
plt.title('Drawdown Curve')
plt.xlabel('Time')
plt.ylabel('Drawdown')
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nPerformance report completed!")

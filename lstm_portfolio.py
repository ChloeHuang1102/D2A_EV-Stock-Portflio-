#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM-Robust投资组合系统
完整的Predict-Then-Optimize框架
"""

# 导入必要的库
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

print("库导入完成")

# ==========================================
# 1. 历史数据获取
# ==========================================

def get_ev_data(n_stocks=30, start_date='2018-01-01', end_date='2024-01-01'):
    """获取EV股票历史数据"""
    print("="*50)
    print("1. 历史数据获取")
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
                print(f"✓ {symbol}: {len(data)} 个交易日")
            else:
                print(f"✗ {symbol}: 数据不足")
        except Exception as e:
            print(f"✗ {symbol}: 错误 - {e}")
    
    raw_data = pd.DataFrame(stock_data).dropna()
    returns = raw_data.pct_change().dropna()
    
    print(f"\n数据获取完成: {len(successful_stocks)} 支股票, {raw_data.shape[0]} 个交易日")
    return raw_data, returns, successful_stocks

# 获取数据
raw_data, returns, stock_names = get_ev_data()

# ==========================================
# 2. 数据预处理
# ==========================================

print("="*50)
print("2. 数据预处理")
print("="*50)

# 标准化价格数据
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(raw_data)
scaled_prices = pd.DataFrame(scaled_prices, columns=raw_data.columns, index=raw_data.index)

print(f"价格数据标准化完成: {scaled_prices.shape}")
print(f"收益率数据形状: {returns.shape}")
print("数据预处理完成")

# ==========================================
# 3. 数据分割
# ==========================================

print("="*50)
print("3. 数据分割")
print("="*50)

# 分割参数
test_size = 30
sequence_length = 20

# 分割数据
train_data = scaled_prices.iloc[:-test_size]
test_data = scaled_prices.iloc[-test_size:]

print(f"训练集: {train_data.shape}")
print(f"测试集: {test_data.shape}")
print(f"序列长度: {sequence_length}")
print("数据分割完成")

# ==========================================
# 4. LSTM模型训练
# ==========================================

print("="*50)
print("4. LSTM模型训练")
print("="*50)

def prepare_lstm_data(data, seq_length):
    """准备LSTM训练数据"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data.iloc[i-seq_length:i].values)
        y.append(data.iloc[i].values)
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, n_stocks):
    """构建LSTM模型"""
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

# 准备训练数据
X_train, y_train = prepare_lstm_data(train_data, sequence_length)
X_val, y_val = prepare_lstm_data(test_data, sequence_length)

# 构建和训练模型
lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]), len(stock_names))

print(f"训练数据形状: {X_train.shape}")
print(f"验证数据形状: {X_val.shape}")
print("开始训练LSTM模型...")

# 训练模型
history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

print("LSTM模型训练完成")

# ==========================================
# 5. 滚动窗口预测
# ==========================================

print("="*50)
print("5. 滚动窗口预测")
print("="*50)

def rolling_prediction(model, test_data, seq_length):
    """滚动窗口预测"""
    predictions = []
    
    for i in range(seq_length, len(test_data)):
        # 获取窗口数据
        window_data = test_data.iloc[i-seq_length:i]
        X_input = window_data.values.reshape(1, seq_length, -1)
        
        # 预测
        pred = model.predict(X_input, verbose=0)
        predictions.append(pred[0])
    
    return np.array(predictions)

# 执行滚动预测
predictions = rolling_prediction(lstm_model, test_data, sequence_length)

print(f"预测结果形状: {predictions.shape}")
print("滚动窗口预测完成")

# ==========================================
# 6. 不确定性量化
# ==========================================

print("="*50)
print("6. 不确定性量化")
print("="*50)

def quantify_uncertainty(model, test_data, seq_length, n_samples=10):
    """量化预测不确定性"""
    predictions_list = []
    
    for i in range(seq_length, len(test_data)):
        window_data = test_data.iloc[i-seq_length:i]
        X_input = window_data.values.reshape(1, seq_length, -1)
        
        # 多次预测
        pred_samples = []
        for _ in range(n_samples):
            pred = model.predict(X_input, verbose=0)
            pred_samples.append(pred[0])
        
        pred_samples = np.array(pred_samples)
        mean_pred = np.mean(pred_samples, axis=0)
        uncertainty = np.std(pred_samples, axis=0)
        
        predictions_list.append((mean_pred, uncertainty))
    
    return predictions_list

# 量化不确定性
uncertainty_results = quantify_uncertainty(lstm_model, test_data, sequence_length)

mean_predictions = np.array([result[0] for result in uncertainty_results])
uncertainties = np.array([result[1] for result in uncertainty_results])

print(f"平均预测形状: {mean_predictions.shape}")
print(f"不确定性形状: {uncertainties.shape}")
print("不确定性量化完成")

# ==========================================
# 7. Robust优化
# ==========================================

print("="*50)
print("7. Robust优化")
print("="*50)

def robust_optimization(expected_returns, uncertainty, cov_matrix, risk_aversion=1.0, uncertainty_factor=0.1):
    """Robust Mean-Variance优化"""
    n = len(expected_returns)
    weights = cp.Variable(n)
    
    # 考虑不确定性的鲁棒收益
    robust_returns = expected_returns - uncertainty_factor * uncertainty
    
    # 目标函数
    portfolio_return = cp.sum(cp.multiply(weights, robust_returns))
    portfolio_risk = cp.quad_form(weights, cov_matrix)
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
    
    # 约束条件
    constraints = [
        cp.sum(weights) == 1,  # 权重和为1
        weights >= 0,  # 不允许做空
        weights <= 0.1,  # 单只股票最大10%
    ]
    
    # 求解
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if problem.status == cp.OPTIMAL:
        return weights.value
    else:
        return np.ones(n) / n  # 等权重

print("Robust优化函数定义完成")

# ==========================================
# 8. 投资组合权重
# ==========================================

print("="*50)
print("8. 投资组合权重")
print("="*50)

# 计算协方差矩阵
cov_matrix = returns.cov().values

# 为每个预测时点计算最优权重
portfolio_weights = []

for i in range(len(mean_predictions)):
    expected_returns = mean_predictions[i]
    uncertainty = uncertainties[i]
    
    # Robust优化
    optimal_weights = robust_optimization(expected_returns, uncertainty, cov_matrix)
    portfolio_weights.append(optimal_weights)

portfolio_weights = np.array(portfolio_weights)

print(f"投资组合权重形状: {portfolio_weights.shape}")
print(f"权重和: {portfolio_weights.sum(axis=1)[:5]}...")
print("投资组合权重计算完成")

# ==========================================
# 9. 回测评估
# ==========================================

print("="*50)
print("9. 回测评估")
print("="*50)

# 计算投资组合收益率
portfolio_returns = []
actual_returns = returns.iloc[-len(portfolio_weights):].values

for i, weights in enumerate(portfolio_weights):
    if i < len(actual_returns):
        portfolio_return = np.dot(weights, actual_returns[i])
        portfolio_returns.append(portfolio_return)

portfolio_returns = np.array(portfolio_returns)

# 计算性能指标
total_return = (1 + portfolio_returns).prod() - 1
volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

# 最大回撤
cumulative = (1 + portfolio_returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()

print(f"投资组合收益率形状: {portfolio_returns.shape}")
print("回测评估完成")

# ==========================================
# 10. 性能报告
# ==========================================

print("="*60)
print("10. 性能报告")
print("="*60)

print(f"投资标的: {len(stock_names)} 支股票")
print(f"测试期间: {len(portfolio_returns)} 个交易日")
print(f"序列长度: {sequence_length} 天")

print("\n" + "-"*60)
print("性能指标")
print("-"*60)
print(f"• 总收益率: {total_return:.4f} ({total_return*100:.2f}%)")
print(f"• 年化波动率: {volatility:.4f} ({volatility*100:.2f}%)")
print(f"• 夏普比率: {sharpe_ratio:.4f}")
print(f"• 最大回撤: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")

print("\n" + "-"*60)
print("模型特点")
print("-"*60)
print("• LSTM预测: 捕捉时间序列模式")
print("• 不确定性量化: 考虑预测风险")
print("• Robust优化: 最大化鲁棒收益")
print("• 风险控制: 限制集中度风险")

# 可视化结果
plt.figure(figsize=(15, 10))

# 投资组合价值曲线
plt.subplot(2, 2, 1)
portfolio_values = (1 + portfolio_returns).cumprod()
plt.plot(portfolio_values, label='投资组合价值', linewidth=2)
plt.title('投资组合价值曲线')
plt.xlabel('时间')
plt.ylabel('价值')
plt.legend()
plt.grid(True)

# 收益率分布
plt.subplot(2, 2, 2)
plt.hist(portfolio_returns, bins=20, alpha=0.7, edgecolor='black')
plt.title('投资组合收益率分布')
plt.xlabel('收益率')
plt.ylabel('频次')
plt.grid(True)

# 权重分布
plt.subplot(2, 2, 3)
avg_weights = portfolio_weights.mean(axis=0)
plt.bar(range(len(stock_names)), avg_weights)
plt.title('平均投资权重')
plt.xlabel('股票')
plt.ylabel('权重')
plt.xticks(range(len(stock_names)), stock_names, rotation=45)
plt.grid(True)

# 回撤曲线
plt.subplot(2, 2, 4)
plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
plt.plot(drawdown, color='red', linewidth=1)
plt.title('回撤曲线')
plt.xlabel('时间')
plt.ylabel('回撤')
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n性能报告完成！")

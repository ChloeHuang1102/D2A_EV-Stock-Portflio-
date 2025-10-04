"""
PTO (Predict-Then-Optimize) 投资组合框架
LSTM预测 + Robust Mean-Variance优化
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import cvxpy as cp
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PTO_Portfolio:
    """
    PTO投资组合：LSTM预测 + Robust Mean-Variance优化
    """
    
    def __init__(self, 
                 n_stocks=30,
                 sequence_length=20,
                 initial_capital=1000000,
                 risk_aversion=1.0,
                 uncertainty_factor=0.1):
        """
        初始化PTO投资组合
        """
        self.n_stocks = n_stocks
        self.sequence_length = sequence_length
        self.initial_capital = initial_capital
        self.risk_aversion = risk_aversion
        self.uncertainty_factor = uncertainty_factor
        
        # 模型组件
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        
        # 结果存储
        self.predictions_history = []
        self.weights_history = []
        self.portfolio_returns = []
        
    def build_lstm_model(self, input_shape):
        """
        构建LSTM预测模型
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.n_stocks, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, returns_data):
        """
        准备LSTM训练数据
        """
        scaled_data = self.scaler.fit_transform(returns_data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i])
        
        return np.array(X), np.array(y)
    
    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=100):
        """
        训练LSTM模型
        """
        print("训练LSTM模型...")
        
        self.lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        
        print("LSTM模型训练完成")
        return history
    
    def predict_with_uncertainty(self, X_input):
        """
        LSTM预测并量化不确定性
        """
        # 多次预测量化不确定性
        predictions_list = []
        for _ in range(10):
            pred = self.lstm_model.predict(X_input, verbose=0)
            predictions_list.append(pred)
        
        predictions_array = np.array(predictions_list)
        mean_predictions = np.mean(predictions_array, axis=0)
        uncertainty = np.std(predictions_array, axis=0)
        
        return mean_predictions, uncertainty
    
    def robust_optimization(self, expected_returns, uncertainty, cov_matrix):
        """
        Robust Mean-Variance优化
        """
        n = len(expected_returns)
        weights = cp.Variable(n)
        
        # 考虑不确定性的鲁棒收益
        robust_returns = expected_returns - self.uncertainty_factor * uncertainty
        
        # 目标函数
        portfolio_return = cp.sum(cp.multiply(weights, robust_returns))
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_risk)
        
        # 约束条件
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= 0.1
        ]
        
        # 求解
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == cp.OPTIMAL:
            return weights.value
        else:
            return np.ones(n) / n
    
    def rolling_prediction(self, test_data):
        """
        滚动窗口预测和优化
        """
        print("开始滚动窗口预测...")
        
        predictions = []
        weights = []
        
        for i in range(self.sequence_length, len(test_data)):
            # 获取窗口数据
            window_data = test_data.iloc[i-self.sequence_length:i]
            
            # 准备LSTM输入
            X_input = self.scaler.transform(window_data).reshape(1, self.sequence_length, self.n_stocks)
            
            # 预测
            pred, unc = self.predict_with_uncertainty(X_input)
            
            # 计算协方差矩阵
            cov_matrix = window_data.cov().values
            
            # Robust优化
            optimal_weights = self.robust_optimization(pred[0], unc[0], cov_matrix)
            
            # 存储结果
            predictions.append(pred[0])
            weights.append(optimal_weights)
            
            if i % 20 == 0:
                print(f"进度: {i}/{len(test_data)}")
        
        return predictions, weights
    
    def backtest(self, test_data, predictions, weights):
        """
        回测投资组合
        """
        print("回测投资组合...")
        
        portfolio_returns = []
        
        for i, (pred, weight) in enumerate(zip(predictions, weights)):
            if i < len(test_data):
                actual_returns = test_data.iloc[i].values
                portfolio_return = np.dot(weight, actual_returns)
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
        
        performance_metrics = {
            'Total Return': total_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Portfolio Returns': portfolio_returns
        }
        
        return performance_metrics
    
    def plot_results(self, test_data, predictions, weights, performance_metrics):
        """
        可视化结果
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 预测 vs 实际
        ax1 = axes[0, 0]
        actual_returns = test_data.iloc[:len(predictions)].mean(axis=1)
        predicted_returns = np.array(predictions).mean(axis=1)
        
        ax1.plot(actual_returns.index, actual_returns.values, label='实际收益', alpha=0.7)
        ax1.plot(actual_returns.index, predicted_returns, label='预测收益', alpha=0.7)
        ax1.set_title('预测 vs 实际收益')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 权重分布
        ax2 = axes[0, 1]
        avg_weights = np.mean(weights, axis=0)
        ax2.bar(range(len(avg_weights)), avg_weights)
        ax2.set_title('平均投资权重')
        ax2.set_xlabel('股票')
        ax2.set_ylabel('权重')
        ax2.grid(True, alpha=0.3)
        
        # 3. 累计收益
        ax3 = axes[1, 0]
        portfolio_returns = performance_metrics['Portfolio Returns']
        cumulative_returns = (1 + portfolio_returns).cumprod()
        ax3.plot(cumulative_returns)
        ax3.set_title('投资组合累计收益')
        ax3.set_ylabel('累计收益')
        ax3.grid(True, alpha=0.3)
        
        # 4. 性能指标
        ax4 = axes[1, 1]
        metrics = ['Total Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']
        values = [performance_metrics[metric] for metric in metrics]
        ax4.bar(metrics, values, alpha=0.7)
        ax4.set_title('性能指标')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, performance_metrics):
        """
        生成性能报告
        """
        print("\n" + "="*60)
        print("PTO投资组合性能报告")
        print("="*60)
        
        print(f"投资标的: {self.n_stocks}支股票")
        print(f"序列长度: {self.sequence_length}天")
        print(f"风险厌恶系数: {self.risk_aversion}")
        print(f"不确定性因子: {self.uncertainty_factor}")
        
        print("\n" + "-"*60)
        print("性能指标")
        print("-"*60)
        
        for metric, value in performance_metrics.items():
            if metric != 'Portfolio Returns':
                if isinstance(value, float):
                    print(f"• {metric}: {value:.4f}")
                else:
                    print(f"• {metric}: {value}")
        
        print("\n" + "-"*60)
        print("模型特点")
        print("-"*60)
        print("• LSTM预测：捕捉时间序列模式")
        print("• Robust优化：考虑预测不确定性")
        print("• 滚动窗口：动态更新预测")
        print("• 风险控制：限制集中度风险")
        
        return performance_metrics

# 使用示例
def run_pto_portfolio(returns_data):
    """
    运行PTO投资组合系统
    """
    print("="*60)
    print("PTO (Predict-Then-Optimize) 投资组合")
    print("="*60)
    
    # 初始化PTO投资组合
    pto = PTO_Portfolio(
        n_stocks=30,
        sequence_length=20,
        initial_capital=1000000,
        risk_aversion=1.0,
        uncertainty_factor=0.1
    )
    
    # 数据分割
    train_data = returns_data.iloc[:int(len(returns_data)*0.7)]
    val_data = returns_data.iloc[int(len(returns_data)*0.7):int(len(returns_data)*0.85)]
    test_data = returns_data.iloc[int(len(returns_data)*0.85):]
    
    print(f"数据分割:")
    print(f"• 训练集: {len(train_data)} 个交易日")
    print(f"• 验证集: {len(val_data)} 个交易日")
    print(f"• 测试集: {len(test_data)} 个交易日")
    
    # 准备数据
    X_train, y_train = pto.prepare_data(train_data)
    X_val, y_val = pto.prepare_data(val_data)
    
    # 训练LSTM
    history = pto.train_lstm(X_train, y_train, X_val, y_val, epochs=50)
    
    # 滚动预测
    predictions, weights = pto.rolling_prediction(test_data)
    
    # 回测
    performance_metrics = pto.backtest(test_data, predictions, weights)
    
    # 可视化
    pto.plot_results(test_data, predictions, weights, performance_metrics)
    
    # 生成报告
    pto.generate_report(performance_metrics)
    
    return pto, performance_metrics

if __name__ == "__main__":
    # 使用模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    returns_data = pd.DataFrame(
        np.random.normal(0.001, 0.02, (len(dates), 30)),
        index=dates,
        columns=[f'Stock_{i}' for i in range(30)]
    )
    
    pto, metrics = run_pto_portfolio(returns_data)

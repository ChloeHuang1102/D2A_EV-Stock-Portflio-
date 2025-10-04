"""
LSTM预测模型数据准备
获取和预处理30支EV股票的历史数据
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class EVStockDataPreparer:
    """
    EV股票数据准备类
    """
    
    def __init__(self, n_stocks=30, start_date='2018-01-01', end_date=None):
        """
        初始化数据准备器
        
        Args:
            n_stocks: 股票数量
            start_date: 开始日期
            end_date: 结束日期
        """
        self.n_stocks = n_stocks
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # EV股票列表（实际使用时需要更新为真实股票代码）
        self.ev_stocks = [
            'TSLA', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'F', 'GM', 'FORD',
            'RACE', 'BMWYY', 'VWAGY', 'TM', 'HMC', 'BYDDF', 'BAMXF', 'DMLRY',
            'STLA', 'NSANY', 'HYMTF', 'RIVN', 'LCID', 'F', 'GM', 'FORD',
            'TSLA', 'NIO', 'XPEV', 'LI', 'RIVN'  # 重复以凑够30支
        ]
        
        self.raw_data = None
        self.processed_data = None
        self.returns = None
        
    def get_stock_data(self):
        """
        获取股票数据
        """
        print(f"正在获取{self.n_stocks}支EV股票数据...")
        print(f"时间范围: {self.start_date} 到 {self.end_date}")
        
        stock_data = {}
        successful_stocks = []
        
        for symbol in self.ev_stocks[:self.n_stocks]:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=self.start_date, end=self.end_date)
                
                if not data.empty and len(data) > 1000:  # 至少1000个交易日
                    stock_data[symbol] = data['Close']
                    successful_stocks.append(symbol)
                    print(f"✓ {symbol}: {len(data)} 个交易日")
                else:
                    print(f"✗ {symbol}: 数据不足")
                    
            except Exception as e:
                print(f"✗ {symbol}: 错误 - {e}")
        
        if len(successful_stocks) < 20:
            print(f"警告: 只有{len(successful_stocks)}支股票数据可用，建议增加股票数量")
        
        self.raw_data = pd.DataFrame(stock_data)
        self.raw_data = self.raw_data.dropna()
        
        print(f"\n数据获取完成:")
        print(f"• 成功获取: {len(successful_stocks)} 支股票")
        print(f"• 数据期间: {self.raw_data.shape[0]} 个交易日")
        print(f"• 数据范围: {self.raw_data.index[0].strftime('%Y-%m-%d')} 到 {self.raw_data.index[-1].strftime('%Y-%m-%d')}")
        
        return self.raw_data
    
    def preprocess_data(self):
        """
        数据预处理
        """
        if self.raw_data is None:
            raise ValueError("请先获取股票数据")
        
        print("正在进行数据预处理...")
        
        # 计算收益率
        self.returns = self.raw_data.pct_change().dropna()
        
        # 数据质量检查
        self._check_data_quality()
        
        # 数据标准化
        self._normalize_data()
        
        print("数据预处理完成")
        return self.returns
    
    def _check_data_quality(self):
        """
        检查数据质量
        """
        print("检查数据质量...")
        
        # 检查缺失值
        missing_values = self.returns.isnull().sum()
        if missing_values.sum() > 0:
            print(f"警告: 发现 {missing_values.sum()} 个缺失值")
        
        # 检查异常值
        for stock in self.returns.columns:
            returns = self.returns[stock].dropna()
            outliers = returns[(returns > 0.2) | (returns < -0.2)]
            if len(outliers) > 0:
                print(f"警告: {stock} 发现 {len(outliers)} 个异常值")
        
        # 检查数据长度
        min_length = self.returns.shape[0]
        print(f"数据长度: {min_length} 个交易日")
        
        if min_length < 1000:
            print("警告: 数据长度不足，可能影响LSTM模型训练")
    
    def _normalize_data(self):
        """
        数据标准化
        """
        # 使用Z-score标准化
        self.processed_data = (self.returns - self.returns.mean()) / self.returns.std()
        
        print("数据标准化完成")
    
    def split_data(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        分割数据为训练集、验证集和测试集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        if self.returns is None:
            raise ValueError("请先进行数据预处理")
        
        total_length = len(self.returns)
        train_end = int(total_length * train_ratio)
        val_end = int(total_length * (train_ratio + val_ratio))
        
        train_data = self.returns.iloc[:train_end]
        val_data = self.returns.iloc[train_end:val_end]
        test_data = self.returns.iloc[val_end:]
        
        print(f"数据分割完成:")
        print(f"• 训练集: {len(train_data)} 个交易日 ({train_ratio:.1%})")
        print(f"• 验证集: {len(val_data)} 个交易日 ({val_ratio:.1%})")
        print(f"• 测试集: {len(test_data)} 个交易日 ({test_ratio:.1%})")
        
        return train_data, val_data, test_data
    
    def create_lstm_dataset(self, sequence_length=20):
        """
        创建LSTM训练数据集
        
        Args:
            sequence_length: 序列长度（时间窗口）
        """
        if self.processed_data is None:
            raise ValueError("请先进行数据预处理")
        
        print(f"创建LSTM数据集，序列长度: {sequence_length}")
        
        # 为每支股票创建序列数据
        X, y = [], []
        
        for stock in self.processed_data.columns:
            stock_data = self.processed_data[stock].values
            
            for i in range(sequence_length, len(stock_data)):
                # 输入序列
                X.append(stock_data[i-sequence_length:i])
                # 输出（下一个时间点的收益率）
                y.append(stock_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"LSTM数据集创建完成:")
        print(f"• 样本数量: {len(X)}")
        print(f"• 序列长度: {sequence_length}")
        print(f"• 特征维度: {X.shape[1]}")
        
        return X, y
    
    def plot_data_overview(self):
        """
        绘制数据概览
        """
        if self.raw_data is None:
            raise ValueError("请先获取股票数据")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 价格走势
        ax1 = axes[0, 0]
        sample_stocks = self.raw_data.columns[:5]  # 显示前5支股票
        for stock in sample_stocks:
            ax1.plot(self.raw_data.index, self.raw_data[stock], label=stock, alpha=0.7)
        ax1.set_title('股票价格走势（样本）')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 收益率分布
        ax2 = axes[0, 1]
        if self.returns is not None:
            sample_returns = self.returns[sample_stocks]
            ax2.hist(sample_returns.values.flatten(), bins=50, alpha=0.7, density=True)
            ax2.set_title('收益率分布')
            ax2.set_xlabel('收益率')
            ax2.set_ylabel('密度')
            ax2.grid(True, alpha=0.3)
        
        # 3. 波动率时间序列
        ax3 = axes[1, 0]
        if self.returns is not None:
            volatility = self.returns.rolling(window=20).std()
            ax3.plot(volatility.index, volatility.mean(axis=1), label='平均波动率')
            ax3.set_title('市场波动率时间序列')
            ax3.set_ylabel('波动率')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 数据质量统计
        ax4 = axes[1, 1]
        if self.returns is not None:
            stats = self.returns.describe()
            ax4.bar(range(len(stats.columns)), stats.loc['std'], alpha=0.7)
            ax4.set_title('各股票波动率对比')
            ax4.set_xlabel('股票')
            ax4.set_ylabel('标准差')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_data_summary(self):
        """
        获取数据摘要
        """
        if self.raw_data is None:
            return "请先获取股票数据"
        
        summary = {
            '股票数量': len(self.raw_data.columns),
            '数据长度': len(self.raw_data),
            '时间范围': f"{self.raw_data.index[0].strftime('%Y-%m-%d')} 到 {self.raw_data.index[-1].strftime('%Y-%m-%d')}",
            '数据完整性': f"{(1 - self.raw_data.isnull().sum().sum() / (len(self.raw_data) * len(self.raw_data.columns))):.2%}"
        }
        
        if self.returns is not None:
            summary.update({
                '平均日收益率': f"{self.returns.mean().mean():.4f}",
                '平均波动率': f"{self.returns.std().mean():.4f}",
                '最大单日涨幅': f"{self.returns.max().max():.2%}",
                '最大单日跌幅': f"{self.returns.min().min():.2%}"
            })
        
        return summary

# 使用示例
def prepare_ev_data():
    """
    准备EV股票数据的完整流程
    """
    print("="*60)
    print("EV股票数据准备")
    print("="*60)
    
    # 初始化数据准备器
    preparer = EVStockDataPreparer(
        n_stocks=30,
        start_date='2018-01-01',  # 5年数据
        end_date='2024-01-01'
    )
    
    # 获取数据
    raw_data = preparer.get_stock_data()
    
    # 预处理数据
    returns = preparer.preprocess_data()
    
    # 分割数据
    train_data, val_data, test_data = preparer.split_data()
    
    # 创建LSTM数据集
    X, y = preparer.create_lstm_dataset(sequence_length=20)
    
    # 绘制数据概览
    preparer.plot_data_overview()
    
    # 获取数据摘要
    summary = preparer.get_data_summary()
    print("\n数据摘要:")
    for key, value in summary.items():
        print(f"• {key}: {value}")
    
    return raw_data, returns, train_data, val_data, test_data, X, y

if __name__ == "__main__":
    # 运行数据准备
    raw_data, returns, train_data, val_data, test_data, X, y = prepare_ev_data()

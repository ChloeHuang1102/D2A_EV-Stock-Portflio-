# EV Stock Portfolio Optimization Using Smart Predict-then-Optimize (PTO) Framework

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Problem Formulation](#problem-formulation)
3. [Smart PTO Framework](#smart-pto-framework)
4. [Robust Optimization for EV Stocks](#robust-optimization-for-ev-stocks)
5. [SPO+ Loss Function](#spo-loss-function)
6. [Implementation Strategy](#implementation-strategy)
7. [References](#references)

---

## Introduction

This document presents a comprehensive Smart Predict-then-Optimize (PTO) model for electric vehicle (EV) stock portfolio optimization. The approach integrates deep learning prediction with portfolio optimization, specifically addressing the unique challenges of the highly volatile EV stock market.

### Key Contributions
- **LSTM-based Prediction**: Deep learning model for EV stock return forecasting
- **SPO+ Optimization**: Robust portfolio optimization with prediction uncertainty
- **EV Market Focus**: Specialized framework for high-volatility EV stocks
- **End-to-End Learning**: Direct optimization of portfolio performance

---

## Problem Formulation

### Standard Portfolio Optimization
We formulate our EV stock portfolio optimization problem as a mean-variance model with constraints:

$$
\begin{align}
\text{minimize}_{x \in X} \quad & E[\xi^T x] \\
\text{subject to} \quad & x^T \Sigma x \leq \gamma \\
& \sum_{i=1}^{n} x_i \leq 1 \\
& -0.1 \leq x_i \leq 0.1 \quad \forall i = 1, 2, \ldots, n
\end{align}
$$

### Notation
- $x \in \mathbb{R}^n$: Decision vector representing portfolio weights for EV stocks
- $\xi \in \mathbb{R}^n$: Loss vector (negative returns) for EV stocks
- $\Sigma$: Covariance matrix of EV stock returns
- $\gamma$: Risk threshold parameter
- **Constraints**: Limited exposure (±10% per stock) and no leverage

---

## Smart PTO Framework

### Core Concept
The Smart PTO framework addresses the fundamental challenge: **predictions are uncertain, but optimization requires deterministic inputs**.

### Mathematical Reformulation
Following Elmachtoub and Grigas (2022), we rewrite the problem as:

$$\min_{x \in S} E_{\xi \sim D_w}[\xi^T x|w] = \min_{x \in S} E_{\xi \sim D_w}[\xi|w]^T x$$

where:
- $x \in \mathbb{R}^n$: Decision vector of EV stock weights
- $S \subseteq \mathbb{R}^n$: Feasible set defined by portfolio constraints
- $\xi \in \mathbb{R}^n$: Loss vector (not available at decision time)
- $w \in \mathbb{R}^p$: Observable feature vector (EV market conditions)
- $D_w$: Conditional probability distribution of $\xi$ given $w$

### Two-Step Process

#### Step 1: Predict
Estimate the conditional expectation of EV stock losses:
$$\hat{\xi} = E_{\xi \sim D_w}[\xi|w]$$

#### Step 2: Optimize
Solve the deterministic optimization problem:
$$\min_{x \in S} \hat{\xi}^T x$$

**Key Insight**: $\xi$ represents actual EV stock losses, $\hat{\xi}$ is the predicted value, and $x^*(\hat{\xi})$ gives the optimal EV asset weights.

---

## Robust Optimization for EV Stocks

### Why Robust Optimization for EV Stocks?

EV stocks present unique challenges that require specialized optimization approaches:

#### EV Market Characteristics
1. **Extreme Volatility**: Significantly higher volatility than traditional automotive stocks
2. **Technology Disruption Risk**: Susceptible to battery technology breakthroughs
3. **Regulatory Sensitivity**: Highly sensitive to policy changes and subsidies
4. **Market Sentiment**: Strong influence from social media and news cycles

#### Traditional Approach Limitations
Standard deterministic optimization models fail to account for:
- **Prediction Uncertainty**: EV stock forecasting is inherently challenging
- **Model Misspecification**: Rapidly evolving market dynamics
- **Tail Risk**: Extreme price movements common in EV sector

### Box-Constrained Robust Optimization

#### Uncertainty Set Definition
We assume the true loss vector $\xi$ lies within a box-constrained uncertainty set:

$$\mathcal{U} = \{\xi : \xi_i \in [\hat{\xi}_i - \delta_i, \hat{\xi}_i + \delta_i], i = 1,2,...,n\}$$

where $\delta_i$ represents the maximum deviation for the loss prediction of the i-th EV stock.

#### Robust Optimization Problem
Our robust optimization problem is formulated as:

$$\min_{x \in S} \max_{\xi \in \mathcal{U}} \xi^T x$$

**Equivalent Formulation:**
$$\min_{x \in S} \sum_{i=1}^n (\hat{\xi}_i + \delta_i \cdot \text{sign}(x_i)) \cdot x_i$$

where $\text{sign}(x_i)$ represents the position direction (+1 for long, -1 for short).

#### Parameter Selection Strategy
- **Conservative Approach**: Use 95th percentile of historical prediction errors
- **Adaptive Approach**: Update $\delta_i$ based on recent prediction accuracy
- **Risk-Adjusted**: Scale uncertainty parameters by stock-specific volatility

**Trade-off**: Larger $\delta_i$ → More conservative portfolios (miss gains) vs. Smaller $\delta_i$ → Less protection (higher risk)

---

## SPO+ Loss Function

### The Challenge: Non-Differentiable Loss

The standard SPO loss measures the **surplus cost due to suboptimality**:

$$\ell_{SPO}(\hat{\xi}, \xi) := \max_{x \in X^*(\hat{\xi})} \{\xi^T x\} - z^*(\xi)$$

where $X^*(\hat{\xi}) := \arg\min_{x \in S}\{\hat{\xi}^T x\}$ denotes the set of optimal solutions.

**Problem**: This loss function is **not differentiable**, making gradient-based training impossible.

### SPO+ Solution: Convex Upper Bound

We use the SPO+ loss function as a **convex upper bound**:

$$\ell_{SPO+}(\hat{\xi}, \xi) := \max_{x \in S} \{\xi^T x - 2\hat{\xi}^T x\} + 2\hat{\xi}^T x^*(\hat{\xi}) - z^*(\xi)$$

### Key Properties

1. **Upper Bound**: $\ell_{SPO}(\hat{\xi}, \xi) \leq \ell_{SPO+}(\hat{\xi}, \xi)$ for all $\hat{\xi} \in \mathbb{R}^n$
2. **Convexity**: $\ell_{SPO+}(\hat{\xi}, \xi)$ is convex in $\hat{\xi}$
3. **Differentiability**: Enables gradient-based optimization

### Gradient Computation

The gradient of the SPO+ loss function is:

$$2(x^*(\xi) - x^*(2\hat{\xi} - \xi)) \in \frac{\partial\ell_{SPO+}(\hat{\xi},\xi)}{\partial\hat{\xi}}$$

**Key Insight**: This gradient enables training our EV portfolio model using standard gradient descent algorithms.

---

## Implementation Strategy

### Four-Step Implementation Process

#### 1. Data Collection
- **Historical Data**: EV stock prices and returns
- **Market Factors**: Technical indicators, volatility measures
- **Feature Engineering**: Moving averages, RSI, momentum indicators

#### 2. Deep Learning Model
- **Architecture**: LSTM neural network for sequential prediction
- **Input**: 30-day sequences of technical indicators
- **Output**: Predicted stock returns for all EV stocks
- **Training**: Standard MSE loss for initial training

#### 3. SPO+ Integration
- **Loss Function**: Replace MSE with SPO+ loss
- **Optimization**: End-to-end training with portfolio optimization
- **Gradient Flow**: Direct optimization of portfolio performance

#### 4. Performance Evaluation
- **Backtesting**: Historical performance analysis
- **Risk Metrics**: Sharpe ratio, maximum drawdown, volatility
- **Comparison**: Against traditional prediction-optimization approaches

### Technical Implementation Details

#### LSTM Architecture
```
Input: (sequence_length, n_features)
├── LSTM Layer 1: 128 units + BatchNorm + Dropout
├── LSTM Layer 2: 64 units + BatchNorm + Dropout  
├── LSTM Layer 3: 32 units + BatchNorm + Dropout
├── Dense Layer 1: 64 units + BatchNorm + Dropout
├── Dense Layer 2: 32 units + Dropout
└── Output: n_stocks (linear activation)
```

#### Optimization Constraints
- **Weight Bounds**: $0 \leq x_i \leq 0.1$ (max 10% per stock)
- **Budget Constraint**: $\sum_{i=1}^n x_i = 1$ (fully invested)
- **Risk Management**: Covariance-based risk penalty

---

## References

1. **Zhang, W., Hu, Z., & Yin, H. (2023)**. End-to-End Portfolio Selection: Leveraging Deep Learning in Smart Predict then Optimize. *School of Economics and Management, Tongji University*.

2. **Elmachtoub, A. N., & Grigas, P. (2022)**. Smart "predict, then optimize". *Management Science*, 68(1), 9-26.

---

## Summary

This document presents a comprehensive framework for EV stock portfolio optimization that addresses the unique challenges of high-volatility markets through:

- **Smart PTO Integration**: Seamless combination of prediction and optimization
- **Robust Optimization**: Box-constrained uncertainty sets for EV market volatility
- **SPO+ Loss Function**: Differentiable loss for end-to-end learning
- **LSTM Architecture**: Deep learning for sequential stock return prediction

The framework provides a theoretically sound and practically implementable solution for EV stock portfolio optimization in the presence of prediction uncertainty.
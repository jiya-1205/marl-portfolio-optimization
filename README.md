# Multi-Agent Reinforcement Learning for Algorithmic Execution and Portfolio Optimization

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14+-orange.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0+-green.svg)](https://gymnasium.farama.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.27+-purple.svg)](https://pettingzoo.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**Department of Cyber Physical Systems & School of Electrical Engineering**  
**Manipal Institute of Technology, Manipal Academy of Higher Education (MAHE)**

---

## 📌 Abstract & Motivation

Financial markets are complex, dynamic, and highly sensitive to transaction costs, market impact, and investor behavior. Conventional algorithmic approaches treat trading primarily as a supervised price prediction task. However, accurate price forecasting alone does not yield optimal trading decisions because an automated system must also determine **what capital allocation to assign** (portfolio optimization) and **how orders should be sliced and executed** (algorithmic execution) under market frictions.

This project implements an end-to-end **Multi-Agent Reinforcement Learning (MARL)** framework that decouples and coordinates these two interdependent tasks:
1. **Strategic Portfolio Optimization Agent**: Operates on a macro timescale to determine target capital allocations across a diversified multi-asset basket on the portfolio simplex.
2. **Tactical Trade Execution Agent**: Operates on a micro timescale using the **Almgren-Chriss** market impact formulation to slice rebalancing blocks, minimizing bid-ask spread costs, price impact slippage, and implementation shortfall.

---

## 🏛️ System Architecture

```
                                  +---------------------------------------+
                                  |         Historical Market Data        |
                                  |     (OHLCV + Macro + Tech Indicators) |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |          Feature Engineering          |
                                  | (RSI, MACD, BB, ATR, Return Moments)  |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                           +-----------------------------------------------------+
                           |            Financial Simulation Engine              |
                           |   (Transaction Fees + Bid-Ask Spread + Slippage)    |
                           +--------------------------+--------------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
          +-----------------------------+                           +-----------------------------+
          |  Portfolio Optimization     |                           |       Trade Execution       |
          |           Agent             |                           |            Agent            |
          | (Continuous Simplex Action) |====== Target Rebalance ===>|   (Order Slicing / TWAP)    |
          |       [PPO / SAC]           |<===== Execution Costs =====|       [PPO / SAC]           |
          +--------------+--------------+                           +--------------+--------------+
                         |                                                         |
                         +----------------------------+----------------------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      |     Backtesting & Regimes     |
                                      | (Sharpe, MDD, Bull/Bear/Vol)  |
                                      +---------------+---------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      |  Streamlit Interactive App    |
                                      +-------------------------------+
```

---

## 🔬 Mathematical Formulations

### 1. Portfolio Optimization Formulation
The strategic agent observes state $s_t^{port} \in \mathcal{S}_{port}$ and outputs continuous target portfolio weights $\mathbf{w}_t \in \Delta^{N+1}$:
$$\mathbf{w}_t = \text{Softmax}(\mathbf{a}_t), \quad \sum_{i=1}^N w_{t,i} + w_{t,cash} = 1, \quad w_{t,i} \ge 0$$

The risk-aware reward function penalizes downside variance, turnover friction, and drawdown:
$$r_t^{port} = R_{t+1} - \frac{\lambda_{var}}{2}\max(0, -R_{t+1})^2 - \lambda_{turn} \|\mathbf{w}_{t} - \mathbf{w}_{t-1}\|_1 - \lambda_{dd} D_t^2$$

### 2. Almgren-Chriss Optimal Execution
Given an order of $Q_0$ shares from the strategic agent, the execution agent slices trades across $K$ sub-intervals. The effective transaction price $P_{eff}$ accounts for half-spread and non-linear temporary price impact:
$$P_{eff} = P_0 \pm \left( \text{Spread}_{half} + \gamma \cdot \sigma_{daily} \left(\frac{|q_k|}{V_k}\right)^\alpha \right)$$

The execution reward minimizes Implementation Shortfall (IS) and inventory variance risk:
$$r_k^{exec} = - \frac{|P_{eff} - P_0|}{P_0} - \lambda_{risk} \sigma^2 \left(\frac{q_{rem}}{Q_0}\right)^2 - \text{Fee}_{rate}$$

---

## 📂 Repository Structure

```
minor/
├── data/
│   ├── fetch_data.py          # Market data acquisition & disk caching (yfinance + fallback)
│   ├── feature_pipeline.py    # Technical indicators (RSI, MACD, BB, ATR) & normalization
│   └── cache/                 # Local CSV cache for reproducible offline runs
├── environments/
│   ├── market_frictions.py    # Transaction fees, spreads, Almgren-Chriss slippage
│   ├── portfolio_env.py       # Gymnasium single-agent portfolio optimization
│   ├── execution_env.py       # Gymnasium single-agent algorithmic order slicing
│   └── multi_agent_env.py     # PettingZoo ParallelEnv multi-agent coordination
├── models/
│   ├── networks.py            # Gaussian policy & Twin-Q PyTorch neural networks
│   ├── ppo.py                 # Proximal Policy Optimization (GAE + Clipped objective)
│   └── sac.py                 # Soft Actor-Critic (Twin Q + Max Entropy)
├── baselines/
│   ├── buy_and_hold.py        # Equal-weighted Buy-and-Hold benchmark
│   ├── equal_weight.py        # Periodic 1/N rebalancing benchmark
│   └── markowitz.py           # Modern Portfolio Theory (Mean-Variance optimization)
├── backtesting/
│   ├── metrics.py             # Sharpe, Sortino, Calmar, MDD, Volatility, Turnover
│   ├── regime_analysis.py     # Market regime detector (Bullish, Bearish, High-Vol)
│   └── engine.py              # Out-of-sample test runner & tear-sheet generator
├── app/
│   └── app.py                 # Streamlit interactive visual analytics dashboard
├── checkpoints/               # Model weights (.pt files)
├── results/                   # Backtest output CSVs and comparative figures
├── train.py                   # Unified CLI training script
├── evaluate.py                # Out-of-sample benchmarking runner
├── requirements.txt           # Python dependency specifications
└── README.md                  # Project documentation
```

---

## 🚀 Installation & Quick Start

### 1. Setup Environment
```bash
# Clone or navigate to the project directory
cd c:\Users\Admin\Desktop\minor

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Train Agents
You can train individual agents or the cooperative multi-agent system:

```bash
# Train Multi-Agent System (Portfolio + Execution agents)
python train.py --mode multiagent --episodes 30

# Train single PPO Portfolio Agent
python train.py --mode portfolio --algo ppo --episodes 30

# Train single SAC Portfolio Agent
python train.py --mode portfolio --algo sac --episodes 30

# Train Algorithmic Execution Agent
python train.py --mode execution --episodes 50
```

### 3. Run Out-of-Sample Backtesting & Benchmarks
Execute the full comparative evaluation against **Buy & Hold**, **Equal Weight (1/N)**, and **Markowitz MPT**:

```bash
python evaluate.py
```
This prints the metrics comparison table, regime robustness breakdown, and saves high-resolution charts to `results/backtest_comparison.png`.

### 4. Launch Interactive Web Dashboard
Run the visual Streamlit dashboard:

```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501` to explore:
- **Welcome & Overview**: Visual storytelling of the two-agent cooperative workflow.
- **How the AI Works**: 6-step conceptual journey explaining observation, decision, order slicing, and learning.
- **Implementation Overview**: Honest audit of implemented and verified repository components.
- **Interactive Simulation Lab**: Run real historical market scenarios, inspect the Portfolio Agent's target allocations, and watch the Execution Agent slice orders to minimize price impact.
- **Training & System Status**: Live checkpoint integrity verification and PyTorch inference dry-run status.

---

## 📊 Evaluation Metrics

The framework evaluates all agents and baselines using standard quantitative finance metrics:

| Metric | Description | Formula |
|---|---|---|
| **Cumulative Return** | Total percentage portfolio growth | $\frac{V_T - V_0}{V_0} \times 100\%$ |
| **CAGR** | Compound Annual Growth Rate | $\left(\frac{V_T}{V_0}\right)^{\frac{252}{T}} - 1$ |
| **Annualized Volatility** | Standard deviation of daily returns scaled annually | $\sigma_{daily} \times \sqrt{252}$ |
| **Sharpe Ratio** | Excess return per unit of total risk | $\frac{\mathbb{E}[R - R_f]}{\sigma_{portfolio}} \times \sqrt{252}$ |
| **Sortino Ratio** | Excess return per unit of downside risk | $\frac{\mathbb{E}[R - R_f]}{\sigma_{downside}} \times \sqrt{252}$ |
| **Max Drawdown (MDD)** | Peak-to-trough maximum observed decline | $\max_{t} \left( \frac{\text{Peak}_t - V_t}{\text{Peak}_t} \right)$ |
| **Calmar Ratio** | Return relative to downside tail risk | $\frac{\text{CAGR}}{\text{Max Drawdown}}$ |
| **Frictional Cost Drag** | Total dollars lost to commissions, spread, and slippage | $\sum (\text{Fee} + \text{Spread} + \text{Slippage})$ |

---

## 🎓 Academic Attribution
- **Institution**: Manipal Institute of Technology (MAHE)
- **Specialization**: Minor in Computational Intelligence
- **Authors**: Dhairya Sisodia (230961110), Yash Prasad (230961038), Jiya Srivastava (230961016)
- **Supervisor**: Dr. Vijayalaxmi, School of Electrical Engineering

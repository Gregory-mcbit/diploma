# Multi-Agent AI Investment System

An intelligent investment decision-support system built on a multi-agent architecture. The project combines the capabilities of modern large language models (LLMs) with classical machine learning and financial optimization methods.

## 🌟 Core Concept

Unlike simple chatbots, this system does not "guess" market prices. It is a **Hybrid AI** system in which:
- **LLM (Reasoning)** handles logic, news interpretation, and compliance with the investor's rules.
- **ML & Math (Precision)** handle calculations, identify statistical patterns, and minimize risk.
- **Agents** are independent, specialized modules that "deliberate" with one another to build the final portfolio.

## 🛠 Technology Stack and Applications

The project integrates advanced libraries, each serving a specific purpose:

### 1. Intelligence and Orchestration
*   **LangGraph**: The project's "nervous system." It manages complex agent interaction loops, allows the system to return to a previous step (for example, if the Critic rejects a portfolio), and maintains a shared data state.
*   **OpenAI (GPT-4o-mini)**: The system's "brain." It is used not for numerical calculations, but for reasoning, text analysis, report generation, and tool coordination.

### 2. Models and Data
*   **XGBoost**: Used to predict asset alpha (expected excess returns). The model is trained on historical data, including technical indicators and macroeconomic signals.
*   **yfinance**: The primary tool for obtaining real-time market prices and company fundamentals.

### 3. Financial Mathematics
*   **PyPortfolioOpt**: A professional portfolio optimization library. It calculates precise asset weights based on Modern Portfolio Theory and other contemporary methods while enforcing strict constraints (sector limits, cash allocation, etc.).
*   **Pandas / NumPy**: Industry-standard tools for tabular data manipulation and fast matrix computations.

### 4. Memory and Knowledge (RAG)
*   **ChromaDB**: A vector database that stores "institutional knowledge" (investment rules and risk policies) and cached news. This enables agents to quickly retrieve relevant information without making repeated API requests.

### 5. Infrastructure
*   **FastAPI**: A modern web interface through which the system communicates with the outside world, accepting requests for portfolio construction or monitoring.

## 📊 How the System Works (Pipeline)

1.  **Profiling**: An agent analyzes the user's request, including goals, risk profile, and investment horizon.
2.  **Data Collection**: An agent collects market prices, news, and macroeconomic indicators such as the VIX and bond yields.
3.  **Scoring and Market Regime**: The ML model evaluates asset attractiveness, while the Regime Agent determines the current market state (bullish, bearish, or volatile).
4.  **Portfolio Construction**: Based on the scores and market regime, the mathematical optimizer calculates the optimal asset weights.
5.  **Review and Risk Analysis**: The portfolio is evaluated for volatility and concentration. A dedicated Critic Agent can send it back for reconstruction if it does not match the user's profile.
6.  **Explanation**: The system generates a human-readable report explaining each decision.

## 🚀 Current Status

The system is fully functional for on-demand portfolio construction and monitoring. Work is underway on an autonomous scheduler that will automatically review investments at regular intervals in the background.

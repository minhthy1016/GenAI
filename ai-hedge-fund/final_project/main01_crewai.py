# main01.py
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from crewai import Agent, Task, Crew
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from crewai_tools import LlamaIndexTool
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from rl_ex01 import StockTradingEnv, apply_strategy, sharpe_ratio, max_drawdown, cagr, render_viz, render
from rl_ex01 import QLearningAgent, backtest

# Load and preprocess stock data
def load_and_preprocess_data(file_path):
    stock_data = pd.read_csv(file_path)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)

    # Remove commas from Volume column
    stock_data['Volume'] = stock_data['Volume'].astype(str).str.replace(',', '').astype(float)

    # Convert other columns to float
    cols = ['Open', 'High', 'Low', 'Close']
    stock_data[cols] = stock_data[cols].astype(float)

    # Ensure no NaN values exist
    stock_data = stock_data.dropna().reset_index(drop=True)

    return stock_data


# Function to calculate Sharpe Ratio
def sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe Ratio.
    
    Args:
        returns (array-like): Portfolio returns.
        risk_free_rate (float): Risk-free interest rate (default is 0).
        
    Returns:
        float: Sharpe ratio.
    """
    excess_returns = np.array(returns) - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0

# Function to calculate Maximum Drawdown
def max_drawdown(portfolio_value):
    """Calculate Maximum Drawdown (peak to trough percentage loss).
    
    Args:
        portfolio_value (array-like): Portfolio value over time.
        
    Returns:
        float: Maximum drawdown percentage.
    """
    peak = np.maximum.accumulate(portfolio_value)
    drawdowns = (portfolio_value - peak) / peak
    return np.min(drawdowns)

# Function to calculate CAGR (Compound Annual Growth Rate)
def cagr(final_portfolio_value, initial_balance, trading_days=252):
    """Calculate Compound Annual Growth Rate (CAGR).
    
    Args:
        final_portfolio_value (float): Final value of the portfolio.
        initial_balance (float): Initial capital invested.
        trading_days (int): Number of trading days per year (default is 252).
        
    Returns:
        float: CAGR value.
    """
    years = len(final_portfolio_value) / trading_days
    return (final_portfolio_value[-1] / initial_balance) ** (1 / years) - 1 if years > 0 else 0


# Initialize environment and RL model
stock_data = load_and_preprocess_data('/content/tsla_1ytd.csv')
# Apply strateegy
stock_data = apply_strategy(stock_data)
# Initialize the environment and agent
env = StockTradingEnv(stock_data)
rl_model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.00005, gamma=0.995)
rl_model.learn(total_timesteps=10000)

# LLM and LlamaIndex setup
llm_model = ChatOpenAI(model="gpt-4o-mini")
reader = SimpleDirectoryReader(input_files=['/content/Tesla_analysis_Feb2025.pdf'])
docs = reader.load_data()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
query_engine = index.as_query_engine(similarity_top_k=5, llm=llm_model)

query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="TSLA Query Tool",
    description="Use this tool to lookup TSLA data.",
)

# CrewAI setup
researcher = Agent(
    role="Financial Analyst",
    goal="Analyze TSLA stock performance",
    backstory="Focused on uncovering key financial insights.",
    verbose=True,
    allow_delegation=False,
    tools=[query_tool],
    llm=llm_model,
)

writer = Agent(
    role="Content Strategist",
    goal="Craft reports based on financial analysis",
    backstory="Transforms data insights into compelling narratives.",
    llm=llm_model,
    verbose=True,
    allow_delegation=False,
)

task1 = Task(
    description="Conduct TSLA stock performance analysis.",
    expected_output="Detailed bullet-point report.",
    agent=researcher,
)

task2 = Task(
    description="Draft an engaging article on TSLA's stock trends.",
    expected_output="Article with at least 4 paragraphs.",
    agent=writer,
)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=True)
result = crew.kickoff()

# Backtesting the trained agent
obs = env.reset()
total_rewards = 0
# cash_balance = env.initial_cash_balance
cash_balance = env.initial_balance
# stock_balance = env.initial_stock_balance
stock_balance = env.stock_balance
portfolio_value = []
#total_balance_over_time = []

for step in range(env.max_steps):
    action, _states = rl_model.predict(obs)
    obs, reward, done, info = env.step(action)

    # Track portfolio value (cash + stock holdings)
    portfolio_value.append(cash_balance + stock_balance * env.stock_data.iloc[env.current_step]['Close'])
    total_rewards += reward

    if done:
        break




# Final portfolio value
final_portfolio_value = portfolio_value[-1]




# Example usage
if __name__ == "__main__":
    print("######################")
    print(result)
    # State could be [current_price, moving_avg, RSI, MACD, volume] for a stock trading agent.
    agent = QLearningAgent(state_size=5, action_size=3)

    # Backtest the agent on the stock trading environment
    rewards = backtest(agent, env, episodes=50)

    # Calculate performance metrics
    returns = np.array(rewards)  # Simulating returns for the Sharpe Ratio
    print(f"Final portfolio value: {final_portfolio_value:.2f}")
    print(f"Total reward from backtesting: {total_rewards}")
    print(f"Sharpe Ratio: {sharpe_ratio(returns)}")
    # Print evaluation metrics
    max_drawdown = max_drawdown(portfolio_value)
    cagr = cagr(portfolio_value, env.initial_balance)
    print(f"Max Drawdown: {max_drawdown}")
    print(f"CAGR: {cagr * 100:.2f}%")

    # Render the total balance over time
    env.render()

    env.render_viz()


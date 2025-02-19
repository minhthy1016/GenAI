import numpy as np
import pandas as pd
import random
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import gym
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from gym import spaces

# Stock Trading Environment using Reinforcement Learning
class StockTradingEnv(gym.Env):  # Inherit from gym.Env
    metadata = {'render.modes': ['human']}
    def __init__(self, stock_data, initial_balance=10000, transaction_fee=0.0005, max_steps=200):
        super(StockTradingEnv, self).__init__()

        self.stock_data = stock_data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.reset()


    def reset(self):
        self.current_step = 0
        self.cash_balance = self.initial_balance
        self.stock_balance = 0
        self.total_balance = self.initial_balance
        self.done = False
        self.entry_price = None
        self.history = []

        initial_state = self._get_observation()
        return np.array(initial_state, dtype=np.float32)

    def step(self, action):
        current_price = self.stock_data.iloc[self.current_step]['Close']
        reward = 0

        if action == 1:  # Buy
            if self.cash_balance >= current_price:
                # self.stock_balance += 1
                # self.cash_balance -= current_price * (1 + self.transaction_fee)
                # reward = -current_price * (1 + self.transaction_fee)
                num_shares = self.cash_balance // current_price
                self.stock_balance += num_shares
                self.cash_balance -= num_shares * current_price * (1 + self.transaction_fee)
                self.entry_price = current_price
        elif action == 2 :  # Sell
            if self.stock_balance > 0:
                profit = (current_price - self.entry_price) / self.entry_price
                reward = profit * 100  # Reward for profitable trades
                # self.cash_balance += current_price * (1 - self.transaction_fee)
                # reward = current_price * (1 - self.transaction_fee)
                self.cash_balance += self.stock_balance * current_price * (1 - self.transaction_fee)
                self.stock_balance = 0
                self.entry_price = None

        self.current_step += 1
        self.done = self.current_step >= self.max_steps

        next_state = self._get_observation()
        return np.array(next_state, dtype=np.float32), reward, self.done, {}

    def _get_observation(self):
        stock_info = self.stock_data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line']].values

        # Ensure all values are float
        stock_info = stock_info.astype(np.float32)
        obs = np.concatenate([stock_info, [float(self.cash_balance), float(self.stock_balance)]])

        # Check for NaN or infinite values
        if not np.all(np.isfinite(obs)):
            print(f"⚠️ Warning: Invalid values in observation at step {self.current_step}: {obs}")
            obs = np.nan_to_num(obs)  # Replace NaN/Infs with 0

        return obs


    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Cash: {self.cash_balance}, Stocks: {self.stock_balance}")


    def render_viz(self):
        # Visualize the balance history
        balance = [entry[3] for entry in self.history]
        plt.plot(balance)
        plt.xlabel('Steps')
        plt.ylabel('Total Balance')
        plt.title('Stock Trading - Total Balance Over Time')
        plt.show()

# Apply Strategy Indicators like MA, MACD, EMA
def apply_strategy(df):
    short_window, long_window = 10, 50
    df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Buy_Signal'] = (df['Short_MA'] > df['Long_MA']) & (df['RSI'] > 55) & (df['MACD'] > df['Signal_Line'])
    df['Sell_Signal'] = (df['Short_MA'] < df['Long_MA']) & (df['RSI'] < 45) & (df['MACD'] < df['Signal_Line'])
    return df.dropna()

# Q-Learning Agent for stock trading
class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}  # Q-table initialization

    def get_q_value(self, state, action):
        if (tuple(state) not in self.q_table):
            self.q_table[tuple(state)] = [0] * self.action_size  # Initialize Q-values for all actions

        return self.q_table[tuple(state)][action]

    def update_q_value(self, state, action, reward, next_state, done):
        old_q_value = self.get_q_value(state, action)

        if done:
            future_q_value = 0
        else:
            future_q_value = max(self.get_q_value(next_state, a) for a in range(self.action_size))

        new_q_value = old_q_value + self.alpha * (reward + self.gamma * future_q_value - old_q_value)
        self.q_table[tuple(state)][action] = new_q_value

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))  # Explore: select a random action
        else:
            return np.argmax([self.get_q_value(state, a) for a in range(self.action_size)])  # Exploit: select best action

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay epsilon to reduce exploration

# Backtesting Function to evaluate the agent
def backtest(agent, env, episodes=10):
    all_rewards = []

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)  # Select action
            next_state, reward, done, info = env.step(action)  # Perform action in environment
            agent.update_q_value(state, action, reward, next_state, done)  # Update Q-table
            state = next_state  # Update state
            total_reward += reward  # Accumulate reward

        all_rewards.append(total_reward)

    return all_rewards

# Sharpe Ratio Calculation
def sharpe_ratio(returns, risk_free_rate=0.0):
    return np.mean(returns) / np.std(returns)

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


# Step 3: Initialize the environment
# Load and preprocess stock data (replace with your file path)
stock_data = load_and_preprocess_data('/content/tsla_1ytd.csv')
# Apply strateegy
stock_data = apply_strategy(stock_data)
# Initialize the environment and agent
env = StockTradingEnv(stock_data)

# Step 4: Initialize the RL agent (PPO)
# rl_model = PPO("MlpPolicy", env, verbose=1)
# rl_model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, gamma=0.995)
rl_model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.00005, gamma=0.995)


# Step 5: Train the agent
# rl_model.learn(total_timesteps=10000)
rl_model.learn(total_timesteps=75000)

# Step 6: Backtesting the trained agent
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
print(f"Final portfolio value: {final_portfolio_value:.2f}")
print(f"Total reward from backtesting: {total_rewards}")


# Calculate Max Drawdown (peak to trough percentage loss)
peak = np.maximum.accumulate(portfolio_value)
drawdowns = (portfolio_value - peak) / peak
max_drawdown = np.min(drawdowns)

# Calculate CAGR (Compound Annual Growth Rate)
years = len(stock_data) / 252  # Assuming 252 trading days per year
cagr = (final_portfolio_value / env.initial_balance) ** (1 / years) - 1



# Example usage
if __name__ == "__main__":
    # State could be [current_price, moving_avg, RSI, MACD, volume] for a stock trading agent.
    agent = QLearningAgent(state_size=5, action_size=3)

    # Backtest the agent on the stock trading environment
    rewards = backtest(agent, env, episodes=50)

    # Calculate performance metrics
    returns = np.array(rewards)  # Simulating returns for the Sharpe Ratio
    print(f"Sharpe Ratio: {sharpe_ratio(returns)}")
    # Print evaluation metrics
    print(f"Max Drawdown: {max_drawdown}")
    print(f"CAGR: {cagr * 100:.2f}%")

    # Render the total balance over time
    env.render()

    env.render_viz()

import streamlit as st
import pandas as pd
import PyPDF2
import matplotlib.pyplot as plt
from io import StringIO
from rl_ex01 import load_and_preprocess_data, apply_strategy, StockTradingEnv, backtest, sharpe_ratio
from stable_baselines3 import PPO
import numpy as np


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Streamlit UI
st.title("Stock Analysis & Backtesting Assistant")

st.sidebar.header("Upload Files")
pdf_file = st.sidebar.file_uploader("Upload Stock Analysis PDF", type=["pdf"])
csv_file = st.sidebar.file_uploader("Upload Stock Price CSV", type=["csv"])

if pdf_file:
    st.subheader("Stock Analysis Report")
    extracted_text = extract_text_from_pdf(pdf_file)
    st.text_area("Extracted Text", extracted_text, height=300)

if csv_file:
    st.subheader("Stock Price Data")
    stock_data = load_and_preprocess_data(csv_file)
    stock_data = apply_strategy(stock_data)
    st.dataframe(stock_data.head())

    # Initialize Environment & Model
    env = StockTradingEnv(stock_data)
    rl_model = PPO("MlpPolicy", env, verbose=1)
    rl_model.learn(total_timesteps=10000)

    # Run backtesting
    st.subheader("Backtesting Results")
    agent_rewards = backtest(None, env, episodes=10)  # No Q-learning agent in use
    final_value = env.cash_balance + env.stock_balance * stock_data.iloc[env.current_step]['Close']
    max_drawdown = np.min((np.maximum.accumulate(agent_rewards) - agent_rewards) / np.maximum.accumulate(agent_rewards))
    sharpe = sharpe_ratio(agent_rewards)
    
    st.write(f"**Final Portfolio Value:** ${final_value:.2f}")
    st.write(f"**Max Drawdown:** {max_drawdown:.2%}")
    st.write(f"**Sharpe Ratio:** {sharpe:.2f}")

    # Plot stock price
    st.subheader("Stock Price Trend")
    fig, ax = plt.subplots()
    ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    ax.set_title("Stock Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

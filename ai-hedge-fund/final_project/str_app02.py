import streamlit as st
import pandas as pd
import PyPDF2
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from crew_ex1 import load_and_preprocess_data, apply_strategy, StockMarketEnv, rl_model, crew
from stable_baselines3 import PPO

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Streamlit UI
st.title("AI-Powered Stock Analysis & Backtesting")

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
    env = StockMarketEnv(stock_data)
    rl_model.learn(total_timesteps=10000)

    # Run RL agent for backtesting
    obs = env.reset()
    rewards = []
    for _ in range(10):
        action, _states = rl_model.predict(obs)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    # Run CrewAI for stock analysis
    st.subheader("AI Stock Analysis")
    analysis_result = crew.kickoff()
    st.write(analysis_result)

    # Display backtesting results
    st.subheader("Backtesting Results")
    final_value = env.cash_balance + env.stock_balance * stock_data.iloc[env.current_step]['Close']
    max_drawdown = np.min((np.maximum.accumulate(rewards) - rewards) / np.maximum.accumulate(rewards))
    sharpe = np.mean(rewards) / np.std(rewards)

    st.write(f"**Final Portfolio Value:** ${final_value:.2f}")
    st.write(f"**Max Drawdown:** {max_drawdown:.2%}")
    st.write(f"**Sharpe Ratio:** {sharpe:.2f}")

    # Plot stock price trend
    st.subheader("Stock Price Trend")
    fig, ax = plt.subplots()
    ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    ax.set_title("Stock Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

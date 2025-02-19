# AI Stock Assistant with Reinforcement Learning

## Overview
This project is an AI-powered stock analysis assistant that leverages reinforcement learning (RL) to provide investment insights. The assistant enables users to input U.S. stock names, upload financial reports, and ask investment-related questions, with AI-generated responses based on financial data and market trends.

## Features
- **Stock Analysis with AI**: Utilizes reinforcement learning to analyze stock performance.
- **Financial Data Integration**: Accepts financial sheets and analysis reports for better insights.
- **User-Friendly Interface**: Built using Streamlit for an interactive experience.
- **Real-time Data Retrieval**: Incorporates market data using `alpha_vantage`.

## Dependencies
The project requires the following Python packages:
```
numpy==1.24.0
requests==2.31.0
pydantic==2.9.2
pandas==1.5.3
shimmy>=2.0
stable-baselines3
gym
langgraph==0.2.73
langchain
langchain_openai==0.3.6
langchain_experimental==0.3.4
langchain_community==0.3.17
langsmith
streamlit==1.42.0
alpha_vantage==3.0.0
llama_index==0.12.17
matplotlib
plotly
openai
python_dotenv==1.0.1

```

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the Jupyter notebook AI_stock_assistant_with_RL.ipynb
2. Enter a stock name or upload financial reports or stock price data.
3. Ask investment-related questions, and the AI assistant will provide insights.

## Future Improvements
- Enhance reinforcement learning models for more accurate predictions.
- Enhance more stock data for training and backtesting.
- The AI stock-analyst assistant seems to be performing well in general stock analysis but gives bad decision in real trade (backtesting). It is result from lack of well formatted trainingset, well-organised and excusion between Analyst Agents and Trading Execusion Agents. Moving forwards, we can apply and finetune RL hyperparameter to get signals, suggestions from Analyst Agents to make trade decision and rewards in backtesting. 
- Improve user interface and visualization features.
- Expand support for additional financial APIs.

## Contact
For inquiries or contributions, feel free to reach out to me in my Github profile. 


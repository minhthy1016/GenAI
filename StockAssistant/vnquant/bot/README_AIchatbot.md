# Stock Assistant AI Chatbot

## Overview
The **Stock Assistant AI Chatbot** is a powerful, AI-driven tool designed to assist traders and investors in analyzing stock market trends. Using advanced natural language processing (NLP) and financial analytics, the chatbot offers insightful analysis and actionable recommendations for better trading decisions.

## Key Features
- **Multi-Stock Analysis**: Analyze and compare multiple stocks simultaneously.
- **Technical Indicators**:
  - RSI (Relative Strength Index) for overbought and oversold conditions.
  - MACD (Moving Average Convergence Divergence) for trend analysis.
  - Volatility and percentage changes for daily and monthly performance.
- **Custom Queries**: Ask detailed questions about stock performance, volatility, and trading opportunities.
- **Data-Driven Insights**: Provides clear, up-to-date, and accurate responses with actionable insights.

## SET UP

Follow the steps below to set up and run the project:

1. **Install Dependencies**  
   Run the following command to install the required libraries:  
   ```bash
   pip install -r requirements.txt
   ```

2. **Run get_data.py**
Use this script to crawl stock price data from the vnquant API.
```bash python get_data.py
```
3. **Run data_load.py**
This script processes the crawled stock price data by converting it into the appropriate currency format (e.g., multiplying values by 1,000 VND).
```bash python data_load.py```

4. **Run stock_analysis.py**
This script sets up the Stock Assistant AI Chatbot. Use it to interact with the AI for querying trading opportunities or financial insights.
```bash
python stock_chatbot_analyst.py
```

5. **Reference Notebooks**
Several .ipynb files are provided as reference Jupyter notebooks. Use them for deeper exploration of the code and functionality.

### Key Components
- get_data.py: Crawls stock price data.
- data_load.py: Processes and adjusts stock price data.
- stock_analysis.py: Analyzes and visualizes stock performance.
- stock_chatbot_analyst.py: AI-powered chatbot for stock analysis.
- Notebooks (.ipynb): Detailed step-by-step examples for learning and reference.


## Technologies Used
- **LangChain Community Toolkit**: Enables efficient retrieval-based AI workflows.
- **OpenAI API**: Powered by models like `gpt-3.5-turbo` for natural language understanding.
- **Python Libraries**:
  - **pandas**: Data processing and transformation.
  - **matplotlib**: Visualizing stock performance trends.

## How It Works
1. **Data Loading**: Combines stock data from multiple files into a unified dataset.
2. **Indicator Calculation**: Computes technical indicators (e.g., RSI, MACD, volatility) for all stocks.
3. **Analysis & Insights**: Uses advanced NLP to provide detailed analysis and trading opportunities.
4. **Visualization**: Generates charts for stock performance, RSI, and MACD for better understanding.

## Example Use Case
1. **Question**: "Compare the RSI, MACD, and volatility of APPL, GOOGL, and E1VFVN30 stocks."
2. **Response**: A detailed comparison highlighting trading opportunities for each stock, along with visualizations.

## Future Enhancements
- Integration of live stock data for real-time analysis.
- Portfolio optimization recommendations.
- Sentiment analysis of stock news and market trends.

## Contact
For questions or contributions, feel free to reach out or suggest improvements!

---

**Empower your trading decisions with AI-driven insights!**

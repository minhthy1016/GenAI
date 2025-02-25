# AI Stock Assistant with Reinforcement Learning

## Project Information
**Project Name:** AI Stock Assistant with RL  
**Brief Description:** This project is an AI-powered stock analysis assistant that leverages reinforcement learning (RL) to provide investment insights. Users can input U.S. stock names, upload financial reports, and ask investment-related questions, with AI-generated responses based on financial data and market trends.

## Introduction
The AI Stock Assistant utilizes reinforcement learning to analyze stock performance and provide data-driven investment insights. By integrating financial reports and real-time market data, it offers investors a sophisticated tool for decision-making.

## Features
- **Stock Analysis with AI:** Uses RL to analyze stock trends and suggest insights.
- **Financial Data Integration:** Accepts financial sheets and reports for enhanced predictions.
- **Interactive Interface:** Built with Streamlit for a seamless user experience.
- **Real-time Data Retrieval:** Utilizes `alpha_vantage` for market data.

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the AI_stock_assistant_with_RL.ipynb
2. Enter a stock name or upload financial reports.
3. Ask investment-related questions, and receive AI-generated insights.

## Results
- AI-generated stock performance insights
- Visualizations of financial trends
- Real-time market data analysis

## Technical Information
### Technologies Used
- **Programming Language:** Python
- **Libraries & Frameworks:**
  - `stable-baselines3`, `gym`, `langgraph`, `langchain`, `langchain_openai`
  - `langchain_experimental`, `langsmith`, `pandas`, `streamlit`, `alpha_vantage`
  - `llama-index`, `matplotlib`, `plotly`, `pydantic`, `openai`, `shimmy>=2.0`

### System Requirements
- Python 3.8+
- Internet connection for real-time data fetching
- Compatible with Windows, macOS, and Linux

### Project Structure
```
/AI_Stock_Assistant/
│── AI_stock_assistant_with_RL.ipynb           # Pretrained models notebook
│── str_app.py           # Main application file
│── str_app02.py         # Main application file
│── rl_ex01.py           # RL model test file
│── crew_ex1.py          # crewai app test file
│── main01_crewai.py     # Main application file with RL model
│── requirements.txt     # Dependencies
│── data/                # Directory for financial reports
│── README.md            # Project documentation
```

## Support and Development
### Support
For inquiries, contact:
- Email: minhthy1016@gmail.com
- Community: 
### Development
- **Version 1.0:** Initial release with reinforcement learning integration.
- **Future Plans:**
    - Enhance reinforcement learning models for more accurate predictions.
    - Enhance more stock data for training and backtesting.
    - The AI stock-analyst assistant seems to be performing well in general stock analysis but gives bad decision in real trade (backtesting). It is result from lack of well formatted trainingset, well-organised and excusion between Analyst Agents and Trading Execusion Agents. Moving forwards, we can apply and finetune RL hyperparameter to get signals, suggestions from Analyst Agents to make trade decision and rewards in backtesting.
    - Improve user interface and visualization features.
    - xpand support for additional financial APIs.

## License and Copyright
### License
This project is licensed under the MIT License.

### Copyright
© 2025 AI Stock Assistant. All rights reserved.

## Conclusion
### Acknowledgments
Special thanks to the open-source community for contributions to AI and finance libraries.

### Conclusion
The AI Stock Assistant aims to bridge the gap between AI and investment decision-making, offering a robust tool for traders and analysts. Future enhancements will focus on accuracy, scalability, and user experience.


# stock_chatbot_analyst.py
'''
We only proceed historical price for E1VFVN30, FUEDCMID, ENF. 
The AI-assistant will give opinion based on historical price and Technical analysis.
'''

import pandas as pd
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# Load all stock data files
file_paths = [
    'vnquant/bot/modified_file_E1VFVN30.csv',
    'vnquant/bot/modified_file_FUEDCMID.csv',
    'vnquant/bot/modified_file_ENF.csv'
]

# Step 1: Combine and Process All Data
dataframes = []
for path in file_paths:
    df = pd.read_csv(path)
    df['code'] = path.split("_")[-1].split(".")[0]  # Extract stock code from filename
    df['date'] = pd.to_datetime(df['date'])  # Convert 'date' column to datetime
    dataframes.append(df)

# Combine all datasets
data = pd.concat(dataframes, ignore_index=True)

# Remove duplicate rows if they exist
data = data.drop_duplicates(subset=['date', 'code'], keep='first')

# Set 'date' as index
data.set_index('date', inplace=True)

# Step 2: Process Indicators
# Calculate daily percentage change
data['daily_change'] = (data['close'] - data['open']) / data['open'] * 100

# Calculate monthly percentage change
monthly_change = (
    data.groupby('code')['close']
    .resample('M')
    .ffill()
    .pct_change()
    .reset_index()
    .rename(columns={'close': 'monthly_change'})
)
monthly_change['monthly_change'] = monthly_change['monthly_change'] * 100  # Multiply by 100 after reset_index

# Merge monthly change back to original DataFrame
data = data.reset_index()  # Reset index to make 'date' a column for merging
monthly_change['date'] = pd.to_datetime(monthly_change['date'])  # Ensure correct datetime type
data = data.merge(monthly_change, on=['date', 'code'], how='left').set_index('date')  # Re-set 'date' as index

# Step 3: Calculate RSI
window_length = 14
# Calculate RSI for each stock code separately
data['RSI'] = data.groupby('code')['close'].transform(lambda x: 100 - (100 / (1 + (x.diff().where(x.diff() > 0, 0).rolling(window=window_length).mean() / -x.diff().where(x.diff() < 0, 0).rolling(window=window_length).mean().abs()))))


# Step 4: Calculate MACD and Signal Line
# Calculate ema_12 and ema_26 without groupby, using transform instead
data['ema_12'] = data.groupby('code')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
data['ema_26'] = data.groupby('code')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())

# calculate MACD using the new columns
data['MACD'] = data['ema_12'] - data['ema_26']

data['MACD_signal'] = data.groupby('code')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

# Step 5: Calculate Volatility
data['volatility'] = (data['high'] - data['low']) / data['low'] * 100

# Step 6: Calculate Moving Average
data = data.reset_index()  # Reset the index to have a unique index
data['moving_average'] = data.groupby('code')['close'].rolling(window=7).mean().values
data = data.set_index('date') # Set the index back to 'date'

# Drop NaN values
data.dropna(inplace=True)

# Step 7: Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
documents = [
    Document(page_content=str(row.to_dict()), metadata=row.to_dict()) for _, row in data.iterrows()
]  # Convert dataframe rows to Document objects
vector_store = FAISS.from_documents(documents, embeddings)  # Create FAISS vector store from documents

# Step 8: Define Comparative Analysis Prompt
template = """
You are an Advanced Stock Assistant AI specializing in analyzing multiple stocks. The following data contains insights for {context}.
Provide:
1. A comparative analysis of volatility, RSI, MACD, daily change, and monthly change between all stocks.
2. Identify trading opportunities for each stock.
3. Highlight any stocks with exceptionally high or low trading opportunities based on these indicators.
If unsure, admit that you don't know or suggest contacting an expert.
"""
prompt = PromptTemplate(template=template, input_variables=["context"])

# Step 9: Initialize RetrievalQA Chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

# Step 10: Query the Assistant
query_03 = "Can you provide a comparative analysis of E1VFVN30, APPL, and GOOGL stock trading opportunities?"
response_03 = qa_chain({"query": query_03})
print(response_03['result'])


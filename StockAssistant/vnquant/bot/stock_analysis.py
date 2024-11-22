# stock_analysis.py

import pandas as pd
import matplotlib.pyplot as plt


# Load all stock data files
file_paths = [
    '/content/drive/MyDrive/Colab Notebooks/GenAI course/bossAI/StockAssistantbot/modified_file_E1VFVN30.csv',
    '/content/drive/MyDrive/Colab Notebooks/GenAI course/bossAI/StockAssistantbot/modified_file_FUEDCMID.csv',
    '/content/drive/MyDrive/Colab Notebooks/GenAI course/bossAI/StockAssistantbot/modified_file_ENF.csv'
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

## Visualize all 3 shares
# Plot close price, RSI, and MACD for each stock
stock_codes = data['code'].unique()

for code in stock_codes:
    stock_data = data[data['code'] == code]

    plt.figure(figsize=(14, 8))
    
    # Close Price
    plt.subplot(3, 1, 1)
    plt.plot(stock_data.index, stock_data['close'], label=f'{code} Close Price')
    plt.title(f'{code} Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()

    # RSI
    plt.subplot(3, 1, 2)
    plt.plot(stock_data.index, stock_data['RSI'], label=f'{code} RSI', color='orange')
    plt.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
    plt.title(f'{code} RSI Over Time')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()

    # MACD
    plt.subplot(3, 1, 3)
    plt.plot(stock_data.index, stock_data['MACD'], label=f'{code} MACD', color='blue')
    plt.plot(stock_data.index, stock_data['MACD_signal'], label=f'{code} Signal Line', color='red')
    plt.title(f'{code} MACD and Signal Line Over Time')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()

    plt.tight_layout()
    plt.show()

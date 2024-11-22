import pandas as pd

# Load the CSV file
df = pd.read_csv("vnquant/bot/ENF.csv")

# Multiply all columns by 1000 VND (only if the cell is numeric)
df = df.applymap(lambda x: x * 1000 if isinstance(x, (int, float)) else x)

# Save the modified data back to a new CSV file
df.to_csv("vnquant/bot/modified_file_ENF.csv", index=False)


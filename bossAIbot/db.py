import sqlite3
import pandas as pd

# Load data
employee_data= pd.read_csv('HR/HRDataset_v14.csv')
superstore_sale1= pd.read_csv('Finance/superstore_sales_01.csv')
superstore_sale2= pd.read_csv('Finance/superstore_sale_02.csv')
real_estate_sale = pd.read_csv('Finance/Real_Estate_Sales_2001-2022.csv')
vehicle_sale= pd.read_csv('Finance/asamamove_sales_data.csv')

# Create db connection
employee_data.columns = employee_data.columns.str.strip()
superstore_sale1.columns = superstore_sale1.columns.str.strip()
superstore_sale2.columns = superstore_sale2.columns.str.strip()
real_estate_sale.columns = real_estate_sale.columns.str.strip()
vehicle_sale.columns = vehicle_sale.columns.str.strip()

HR_db_connection = sqlite3.connect('HR_data.db')
Finance_db_connection = sqlite3.connect('Finance_data.db')

# Load to SQLite3 db
employee_data.to_sql('employee_data_attrition',HR_db_connection, if_exists='replace')
superstore_sale1.to_sql('superstore_sale1',Finance_db_connection, if_exists='replace')
superstore_sale2.to_sql('superstore_sale2',Finance_db_connection, if_exists='replace')
real_estate_sale.to_sql('real_estate_sale',Finance_db_connection, if_exists='replace')
vehicle_sale.to_sql('verhicle_sale',Finance_db_connection, if_exists='replace')

print('All is OK')



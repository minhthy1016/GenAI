import os
from langchain_community.utilities.sql_database import SQLDatabase
# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

# Set up OpenAI API Key
openai.api_key = os.environ['OPENAI_API_KEY']
#langchain.api_key = os.environ['LANGCHAIN_API_KEY']

# Define the SQLite database path
db_name = "HR_data.db"  # The database file you created
db_path = f"sqlite:///{db_name}"      # SQLite connection URI format

# Connect to the SQLite database using SQLDatabase from LangChain
db = SQLDatabase.from_uri(
    db_path,
    sample_rows_in_table_info=1,  # Adjust sample rows per table if needed
    include_tables=['employee_data_attrition'],  # Specify tables to include
    custom_table_info={'employee_data_attrition': "Employee details and attrition factors"}
)

# Print database dialect and table information
print("Database Dialect:", db.dialect)
print("Usable Table Names:", db.get_usable_table_names())
print("Table Information:", db.table_info)

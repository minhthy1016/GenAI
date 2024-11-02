import os
import langchain_community
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']


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

# If you build with gpt-3.5-turbo
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
# chain = write_query | execute_query
# chain.invoke({"question": "How many Sale Presentatives and Sale Managers employees are there?"})

# If you build with gpt-4o-mini
#llm = ChatOpenAI(model="gpt-4o-mini")
#chain = create_sql_query_chain(llm, db)
#response = chain.invoke({"question": "How many Sale Presentatives and Sale Managers employees are there?"})
#print(response)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)
user_input_question = input("Enter your QA question\n: ")
#response = chain.invoke({"question": "How many Sale Presentatives and Sale Managers employees are there?"})
response = chain.invoke({"question": user_input_question})
print(response)
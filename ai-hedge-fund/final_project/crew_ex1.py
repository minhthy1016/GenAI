import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from crewai import Agent, Task, Crew
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from crewai_tools import LlamaIndexTool
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI

# Load stock data
stock_data = pd.read_csv('/content/tsla_1ytd.csv')

# Clean numeric data by removing commas and converting to float
for col in stock_data.columns:
    if stock_data[col].dtype == 'object':
        stock_data[col] = stock_data[col].str.replace(',', '', regex=True)
        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')

# Convert date columns to datetime and drop them for RL model
if 'Date' in stock_data.columns:
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
    stock_data = stock_data.drop(columns=['Date'])

# Fill missing values if any
stock_data = stock_data.fillna(0)

# Define Stock Market Environment
class StockMarketEnv(gym.Env):
    def __init__(self, data):
        super(StockMarketEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = np.random.rand()  # Placeholder for actual reward calculation
        obs = self.data.iloc[self.current_step].values.astype(np.float32) if not done else np.zeros(self.data.shape[1], dtype=np.float32)
        return obs, reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Data: {self.data.iloc[self.current_step].to_dict()}")

# Initialize environment and RL model
env = StockMarketEnv(stock_data)
rl_model = PPO("MlpPolicy", env, verbose=1)
rl_model.learn(total_timesteps=10000)

# LLM and LlamaIndex setup
llm_model = ChatOpenAI(model="gpt-4o-mini")
reader = SimpleDirectoryReader(input_files=['/content/Tesla_dbs.pdf'])
docs = reader.load_data()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
query_engine = index.as_query_engine(similarity_top_k=5, llm=llm_model)

query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="TSLA Query Tool",
    description="Use this tool to lookup TSLA data.",
)

# CrewAI setup
researcher = Agent(
    role="Financial Analyst",
    goal="Analyze TSLA stock performance",
    backstory="Focused on uncovering key financial insights.",
    verbose=True,
    allow_delegation=False,
    tools=[query_tool],
    llm=llm_model,
)

writer = Agent(
    role="Content Strategist",
    goal="Craft reports based on financial analysis",
    backstory="Transforms data insights into compelling narratives.",
    llm=llm_model,
    verbose=True,
    allow_delegation=False,
)

task1 = Task(
    description="Conduct TSLA stock performance analysis.",
    expected_output="Detailed bullet-point report.",
    agent=researcher,
)

task2 = Task(
    description="Draft an engaging article on TSLA's stock trends.",
    expected_output="Article with at least 4 paragraphs.",
    agent=writer,
)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=True)
result = crew.kickoff()

# Run RL agent
obs = env.reset()
for _ in range(10):
    action, _states = rl_model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break

print("######################")
print(result)

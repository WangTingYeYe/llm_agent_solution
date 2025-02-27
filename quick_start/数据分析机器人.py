#  初始化大模型
from langchain_community.chat_models import ChatTongyi
import os

from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

model = ChatTongyi(model_name="qwen-turbo")


# Generate sample data
n_rows = 1000

# Generate dates
start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_rows)]

# Define data categories
makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Kia']
models = ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Coupe', 'Van']
colors = ['Red', 'Blue', 'Black', 'White', 'Silver', 'Gray', 'Green']

# Create the dataset
data = {
    'Date': dates,
    'Make': np.random.choice(makes, n_rows),
    'Model': np.random.choice(models, n_rows),
    'Color': np.random.choice(colors, n_rows),
    'Year': np.random.randint(2015, 2023, n_rows),
    'Price': np.random.uniform(20000, 80000, n_rows).round(2),
    'Mileage': np.random.uniform(0, 100000, n_rows).round(0),
    'EngineSize': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
    'FuelEfficiency': np.random.uniform(20, 40, n_rows).round(1),
    'SalesPerson': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eva'], n_rows)
}

# Create DataFrame and sort by date
df = pd.DataFrame(data).sort_values('Date')

# Display sample data and statistics
print("\n生成数据的前几行:")
print(df.head())

print("\n数据信息:")
df.info()

print("\n摘要统计信息:")
print(df.describe())

agent_executor = create_pandas_dataframe_agent(
    model,
    df,
    agent_type="openai-functions",
    verbose=True
)

def ask_agent(question):
    """通过代理提问"""
    response = agent_executor.invoke({
        "input": question,
        "agent_scratchpad": f"Human: {question}\n"
                            f"AI: 为了回答这个问题，我需要使用python来分析数据框架。我将使用python_repl_ast工具.\n\n"
                            f"Action: python_repl_ast\n"
                            f"Action Input: ",
    })
    print(f"Question: {question}")
    print(f"Answer: {response}")
    print("---")


ask_agent("这个数据集中的列名是什么?")
ask_agent("这个数据集中有多少行?")
ask_agent("汽车的平均售价是多少？?")
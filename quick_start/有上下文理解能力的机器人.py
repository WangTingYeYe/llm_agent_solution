from langchain_community.chat_models import ChatTongyi
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


#  初始化大模型
model = ChatTongyi(model_name="qwen-turbo")

# 用于存储聊天历史记录的
store = {}


def get_chat_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助理."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 利用 LCEL 表达式链接创建 Langchain  中的 Chain
chain = prompt | model

# 创建有 历史记录的 chain
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

session_id = "user_123"


response1 = chain_with_history.invoke(
    {"input": "你好我是，伽顿！"},
    config={"configurable": {"session_id": session_id}}
)
print("AI:", response1.content)

response2 = chain_with_history.invoke(
    {"input": "我是谁？"},
    config={"configurable": {"session_id": session_id}}
)
print("AI:", response2.content)

print("\n历史对话:")
for message in store[session_id].messages:
    print(f"{message.type}: {message.content}")
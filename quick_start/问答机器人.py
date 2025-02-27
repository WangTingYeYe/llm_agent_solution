from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate

#  初始化大模型
model = ChatTongyi(model_name="qwen-turbo")
template = """
你是一个AI助理。你的任务是尽你所能回答用户的问题。

用户提问：{question}

请给出一个清晰、简明的答案：
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

#  利用 LCEL表达式构建 langchain 链
qa_chain = prompt | model

def get_answer(question):
    """
    通过qa_chain生产答案
    """
    input_variables = {"question": question}
    response = qa_chain.invoke(input_variables).content
    return response

question = "中国的首都是哪里？"
answer = get_answer(question)
print(f"Question: {question}")
print(f"Answer: {answer}")


user_question = input("输入你的问题：")
user_answer = get_answer(user_question)
print(f"Answer: {user_answer}")
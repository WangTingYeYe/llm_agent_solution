from typing import TypedDict, Annotated, List

from IPython.core.display import Image
from IPython.core.display_functions import display
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph

model = ChatTongyi(model_name="qwen-turbo")


class PlannerState(TypedDict):
    requirement: str
    pre_itinerary_plan: str
    local_itinerary_suggestion: str
    itinerary_precautions: str
    final_itinerary_plan: str


#  1、根据用户的需求创建一个旅行计划
#  2、针对当前的旅行计划，优化一下当地特色
#  3、针对前面的旅行计划，优化下旅行中的注意事项
#  4、总结下前面的旅行计划
pre_plan_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个旅行规划助手. 根据用户要求创建一个旅行计划"),
    MessagesPlaceholder(variable_name="messages")
])
pre_plan_chain = pre_plan_prompt | model | StrOutputParser()

local_prompt = ChatPromptTemplate.from_messages([
    ("system","你是一名本地导游，根据用户的旅行目的地。请给一些本地的特色景点、文化、美食等等本地建议"),
    MessagesPlaceholder(variable_name="messages")
])
local_chain = local_prompt | model | StrOutputParser()

precautions_prompt = ChatPromptTemplate.from_messages([
    ("system","你是以为专业的导游，根据用户的旅行诉求，对本次旅行应该注意的事项给出建议。例如：语言、习俗、文化、路途安全等等"),
    MessagesPlaceholder(variable_name="messages")
])
precautions_chain = precautions_prompt | model | StrOutputParser()

final_plan_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一名专业的旅行规划师，根据上下文的旅行计划、本地建议、注意事项总结出一个完整的旅行计划\n\n"
                          "旅行计划：{pre_itinerary_plan}\n"
                          "本地建议：{local_itinerary_suggestion}\n"
                          "注意事项：{itinerary_precautions}\n"),
])
final_plan_chain = final_plan_prompt | model | StrOutputParser()


def input_requirement(state: PlannerState) -> PlannerState:
    requirement = input("请输入你的旅行需求：")
    return {"requirement": requirement}


def pre_itinerary_plan(state: PlannerState) -> PlannerState:
    pre_plan = pre_plan_chain.invoke({"messages": [HumanMessage(content=state["requirement"])]})
    return {"pre_itinerary_plan", pre_plan}


def local_itinerary_plan(state: PlannerState) -> PlannerState:
    local_suggestion = local_chain.invoke({"messages": [HumanMessage(content=state["requirement"])]})
    return {"local_itinerary_suggestion", local_suggestion}


def precautions_itinerary_plan(state: PlannerState) -> PlannerState:
    local_suggestion = precautions_chain.invoke({"messages": [HumanMessage(content=state["requirement"])]})
    return {"itinerary_precautions", local_suggestion}


def final_itinerary_plan(state: PlannerState) -> PlannerState:
    final_plan = final_plan_chain.invoke(state)
    return {"final_itinerary_plan", final_plan}


workflow = StateGraph(PlannerState)

workflow.add_node("input_requirement", input_requirement)
workflow.add_node("pre_itinerary_plan", pre_itinerary_plan)
workflow.add_node("local_itinerary_plan", local_itinerary_plan)
workflow.add_node("precautions_itinerary_plan", precautions_itinerary_plan)
workflow.add_node("final_itinerary_plan", final_itinerary_plan)

workflow.set_entry_point("input_requirement")
workflow.add_edge("input_requirement", "pre_itinerary_plan")
workflow.add_edge("input_requirement", "local_itinerary_plan")
workflow.add_edge("input_requirement", "precautions_itinerary_plan")
workflow.add_edge("pre_itinerary_plan", "final_itinerary_plan")
workflow.add_edge("local_itinerary_plan", "final_itinerary_plan")
workflow.add_edge("precautions_itinerary_plan", "final_itinerary_plan")
app = workflow.compile()
display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)

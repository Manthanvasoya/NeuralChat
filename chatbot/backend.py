from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

load_dotenv()

# OPENAI MODEL 
#llm = ChatOpenAI(model = 'gpt-4o-mini')

# HUGGINGFACE MODLE 
model = HuggingFaceEndpoint(repo_id= 'google/gemma-4-31B-it',
                          task= 'text-generation')
llm = ChatHuggingFace(llm = model)

# state of the workflow
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

# defining the graph
graph = StateGraph(ChatState)
# defining the nodes and edges
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

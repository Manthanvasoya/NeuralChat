from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import asyncio
import threading
from datetime import datetime
from langsmith import traceable



# imports for tools
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun 
from langgraph.prebuilt import ToolNode,tools_condition
import requests

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)

# gemini model
llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview')

''''
# OPENAI MODEL
llm = ChatOpenAI(model = 'gpt-4o-mini')


# HUGGINGFACE MODLE
model = HuggingFaceEndpoint(repo_id= 'meta-llama/Llama-3.1-8B-Instruct',
                          task= 'text-generation')
llm = ChatHuggingFace(llm = model)
'''


# state of the workflow
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    system_instruction = SystemMessage(
        content="You have access to internet search tools. Use them ONLY for: "
                "1) Current events or recent news (last few months), "
                "2) Information that changes frequently. "
                "use get_stock_price tool when user asks for stock prices. "
                "For general knowledge questions (history, basic facts, definitions), answer directly from your training data without using search tools."
    )
    messages_with_system = [system_instruction] + messages
    response = llm.invoke(messages_with_system)
    return {"messages": [response]}



# all tools

# 1.tool for internet search
@tool
def search_internet(query: str) -> str:
    """Search the internet for information using DuckDuckGo"""
    import json
    search_tool = DuckDuckGoSearchRun()
    result = search_tool.run(query)

    # Parse if it's a JSON string
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                # Extract text from each result
                texts = []
                for item in parsed:
                    if isinstance(item, dict) and 'text' in item:
                        texts.append(item['text'])
                    elif isinstance(item, str):
                        texts.append(item)
                return "\n".join(texts) if texts else result
        except (json.JSONDecodeError, TypeError):
            pass

    # If result is a dict/list, convert carefully
    if isinstance(result, (dict, list)):
        if isinstance(result, list):
            texts = []
            for item in result:
                if isinstance(item, dict) and 'text' in item:
                    texts.append(item['text'])
            return "\n".join(texts) if texts else str(result)
        elif 'text' in result:
            return result['text']

    return str(result)

# 2. tool for calculator
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
# 3.tool for fetching any company's stock price

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=MJ6H4KHTJOGKWZZP"
    r = requests.get(url)
    return r.json()     

# binding tools with llm and defining the tool node
tools = [search_internet, calculator, get_stock_price]
llm = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# making connection
conn = sqlite3.connect(database='mydatabase.db',check_same_thread=False)

# Create chat_titles table FIRST - before anything else
def init_database():
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    print("Database initialized successfully")

# Initialize database immediately
init_database()

# Checkpointer
checkpointer = SqliteSaver(conn = conn)

# defining the graph
graph = StateGraph(ChatState)

# defining the nodes and edges
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


# Save chat title to database
def save_chat_title(thread_id, title):
    cursor = conn.cursor()
    thread_id_str = str(thread_id)
    title_str = str(title)

    # Check if thread already exists
    cursor.execute('SELECT created_at FROM chat_titles WHERE thread_id = ?', (thread_id_str,))
    existing = cursor.fetchone()

    if existing:
        # Update only the title, keep the original created_at
        cursor.execute(
            'UPDATE chat_titles SET title = ? WHERE thread_id = ?',
            (title_str, thread_id_str)
        )
    else:
        # Insert new entry with current timestamp
        cursor.execute(
            'INSERT INTO chat_titles (thread_id, title) VALUES (?, ?)',
            (thread_id_str, title_str)
        )
    conn.commit()

# Retrieve all threads with their titles
@traceable(name="get_all_threads",tage=['database','fetch'],metadata={'operation': 'fetch_all_threads from the database with their chatname '})
def get_all_threads():
    all_threads = {}
    try:
        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config['configurable']['thread_id']
            
            cursor = conn.cursor()
            cursor.execute('SELECT title FROM chat_titles WHERE thread_id = ?', (str(thread_id),))
            result = cursor.fetchone()
            
            title = result[0] if result else 'Chat Title'
            all_threads[thread_id] = title
        
        # Sort by creation time (newest first) - get timestamps
        cursor = conn.cursor()
        cursor.execute('SELECT thread_id, created_at FROM chat_titles ORDER BY created_at ASC')
        sorted_threads = cursor.fetchall()
        
        # Create ordered dict with sorted threads
        ordered_threads = {}
        for thread_id, _ in sorted_threads:
            if thread_id in all_threads:
                ordered_threads[thread_id] = all_threads[thread_id]
        
        print(f"Fetched threads: {ordered_threads}")
        return ordered_threads
    except Exception as e:
        print(f"Error fetching threads: {e}")
    
    return all_threads



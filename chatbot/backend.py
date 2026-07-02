from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
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
from pydantic import BaseModel, Field



# imports for tools
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode,tools_condition
import requests

# RAG & PDF Processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import faiss
import pickle
import os
from pathlib import Path
import numpy as np

load_dotenv()

# RAG Configuration
FAISS_BASE_PATH = Path("faiss_indices")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300
TOP_K_CHUNKS = 3

# Ensure faiss_indices directory exists
FAISS_BASE_PATH.mkdir(exist_ok=True)

# Initialize embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

# User Memory Configuration
USER_MEMORY_PATH = FAISS_BASE_PATH / "user_memory"
USER_MEMORY_PATH.mkdir(exist_ok=True)
USER_MEMORY_INDEX_FILE = USER_MEMORY_PATH / "user_memory.faiss"
USER_MEMORY_METADATA_FILE = USER_MEMORY_PATH / "user_memory_metadata.pkl"
TOP_K_MEMORIES = 5  # Retrieve top 5 similar memories

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

# Global context to store current thread_id and user_id for tool access
_CURRENT_THREAD_ID = None
_CURRENT_USER_ID = None  # Persistent user identifier for memory storage


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


# ======================= Chat History Management =======================
# Global setting for how many recent messages to keep
KEEP_RECENT_MESSAGES = 12

def set_keep_recent(count: int):
    """Set the number of recent messages to keep (for UI slider)"""
    global KEEP_RECENT_MESSAGES
    KEEP_RECENT_MESSAGES = max(5, min(50, count))  # Clamp between 5-50

def summarize_messages(messages: list[BaseMessage]) -> str:
    """Create a concise summary of older messages for context."""
    if len(messages) <= KEEP_RECENT_MESSAGES:
        return ""

    old_messages = messages[:-KEEP_RECENT_MESSAGES]
    summary_text = "Previous conversation summary:\n"

    # Extract text from old messages, filtering empty ones
    message_contents = []
    for msg in old_messages:
        content = msg.content
        if isinstance(content, str) and content.strip():
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            message_contents.append(f"{role}: {content}")
        elif isinstance(content, list) and len(content) > 0:
            # Handle list format from Gemini
            text = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
            if text.strip():
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                message_contents.append(f"{role}: {text}")

    if not message_contents:
        return ""

    # Create concise summary
    summary_input = "\n".join(message_contents)
    summary_prompt = f"""Summarize the following conversation history in 2-3 sentences, preserving key facts and decisions:

{summary_input}

Summary:"""

    try:
        summary = llm.invoke(summary_prompt)
        # Handle both string and list formats from different LLMs
        if hasattr(summary, 'content'):
            content = summary.content
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
            summary_text += str(content)
        else:
            summary_text += str(summary)
        return summary_text
    except Exception as e:
        print(f"[WARN] Failed to summarize: {e}")
        return ""


def prepare_messages_with_trimming(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Trim old messages and replace with a summary while respecting Gemini's message ordering.

    Gemini requires: User -> Assistant (with tool) -> Tool Result -> (can insert here)

    Returns:
        - If <= KEEP_RECENT_MESSAGES: return all as-is
        - If > KEEP_RECENT_MESSAGES:
          - Summarize old messages
          - Insert summary after last ToolMessage (respects tool sequence)
          - Append last KEEP_RECENT_MESSAGES messages
    """
    if len(messages) <= KEEP_RECENT_MESSAGES:
        return messages

    from langchain_core.messages import ToolMessage

    old_messages = messages[:-KEEP_RECENT_MESSAGES]
    recent_messages = messages[-KEEP_RECENT_MESSAGES:]

    # Create summary of old messages
    summary_text = summarize_messages(messages)

    if not summary_text:
        # If summary fails, just return recent messages
        return recent_messages

    # Find the last ToolMessage in old messages to place summary after it
    last_tool_idx = -1
    for i in range(len(old_messages) - 1, -1, -1):
        if isinstance(old_messages[i], ToolMessage):
            last_tool_idx = i
            break

    # Build the result with proper ordering
    if last_tool_idx >= 0:
        # Tool messages exist: keep part up to last tool, insert summary, then add recent messages
        summary_msg = HumanMessage(content=summary_text)
        result = (
            old_messages[:last_tool_idx + 1] +  # Messages up to and including last ToolMessage
            [summary_msg] +                      # Insert summary here (respects ordering)
            recent_messages                      # Recent messages (without old messages between)
        )
    else:
        # No tool messages: prepend summary before recent messages
        summary_msg = HumanMessage(content=summary_text)
        result = [summary_msg] + recent_messages

    return result



# ================================= all tools ============================================

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

# making connection
conn = sqlite3.connect(database='mydatabase.db',check_same_thread=False)

# Create chat_titles table FIRST - before anything else
def init_database():
    """Initialize only the chat_titles table. LangGraph's SqliteSaver handles checkpoints & writes."""
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

# Initialize user memory table (for long-term memory)
def init_user_memory_table():
    """Initialize user_memory table for storing user facts and preferences"""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_memory (
            memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            memory_content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            importance_score FLOAT DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    print("User memory table initialized successfully")

init_user_memory_table()

# defining Checkpointer for the graph
checkpointer = SqliteSaver(conn = conn)


# all the helper functions for database operations

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


def delete_chat(thread_id):
    """Delete a chat thread and all its associated checkpoints and messages."""
    try:
        # Delete from checkpointer (removes from checkpoints and writes tables)
        checkpointer.delete_thread(str(thread_id))

        # Delete from chat_titles table (custom metadata)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_titles WHERE thread_id = ?', (str(thread_id),))
        conn.commit()

        print(f"Chat {thread_id} deleted successfully")
        return True
    except Exception as e:
        print(f"Error deleting chat {thread_id}: {e}")
        return False


# ======================= User Memory Functions =======================

def add_user_memory(user_id, memory_type, memory_content):
    """Store a user memory fact with embedding"""
    try:
        # Generate embedding for the memory content
        embedding = embeddings_model.embed_query(memory_content)
        embedding_bytes = np.array(embedding, dtype='float32').tobytes()

        # Store in database
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_memory (user_id, memory_type, memory_content, embedding)
            VALUES (?, ?, ?, ?)
        ''', (str(user_id), memory_type, memory_content, embedding_bytes))
        conn.commit()

        # Update FAISS index
        _update_user_memory_index(user_id)

        return True
    except Exception as e:
        print(f"[WARN] Error adding user memory: {e}")
        return False


def _update_user_memory_index(user_id):
    """Rebuild FAISS index for all user memories"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT memory_id, memory_content, embedding FROM user_memory
            WHERE user_id = ?
        ''', (str(user_id),))
        results = cursor.fetchall()

        if not results:
            return

        # Create FAISS index
        embeddings_list = []
        metadata_list = []

        for memory_id, content, embedding_bytes in results:
            embedding = np.frombuffer(embedding_bytes, dtype='float32').astype('float32')
            embeddings_list.append(embedding)
            metadata_list.append({'memory_id': memory_id, 'content': content})

        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype('float32')
            index = faiss.IndexFlatL2(embeddings_array.shape[1])
            index.add(embeddings_array)

            # Save index and metadata
            faiss.write_index(index, str(USER_MEMORY_INDEX_FILE))
            with open(USER_MEMORY_METADATA_FILE, 'wb') as f:
                pickle.dump({'user_id': user_id, 'metadata': metadata_list}, f)
    except Exception as e:
        print(f"[WARN] Error updating user memory index: {e}")


def retrieve_user_memories(user_id, query, top_k=5):
    """Retrieve most relevant user memories by semantic similarity"""
    try:
        # Check if FAISS index exists
        if not USER_MEMORY_INDEX_FILE.exists():
            return []

        # Load FAISS index and metadata
        index = faiss.read_index(str(USER_MEMORY_INDEX_FILE))
        with open(USER_MEMORY_METADATA_FILE, 'rb') as f:
            data = pickle.load(f)

        # Check if this is the right user
        if data.get('user_id') != str(user_id):
            return []

        # Generate query embedding
        query_embedding = embeddings_model.embed_query(query)
        query_array = np.array([query_embedding], dtype='float32')

        # Search similar memories
        distances, indices = index.search(query_array, min(top_k, len(data['metadata'])))

        memories = []
        for idx in indices[0]:
            if idx < len(data['metadata']):
                memories.append(data['metadata'][idx]['content'])

        return memories
    except Exception as e:
        print(f"[WARN] Error retrieving user memories: {e}")
        return []


def get_memory_context(user_id):
    """Get formatted context string of user memories for injection"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT memory_type, memory_content FROM user_memory
            WHERE user_id = ?
            ORDER BY importance_score DESC, created_at DESC
            LIMIT 5
        ''', (str(user_id),))
        results = cursor.fetchall()

        if not results:
            return ""

        memory_lines = []
        for mem_type, content in results:
            memory_lines.append(f"• {mem_type}: {content}")

        return "\n".join(memory_lines)
    except Exception as e:
        print(f"[WARN] Error getting memory context: {e}")
        return ""


# ======================= Memory Extraction Models =======================

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)


# ======================= Deduplication Logic =======================

def is_memory_duplicate(new_memory_text, user_id, similarity_threshold=0.75):
    """
    Check if a new memory is a duplicate of existing memories using semantic similarity.
    Returns True if duplicate, False if new.
    """
    try:
        # Get existing memories for this user
        cursor = conn.cursor()
        cursor.execute('''
            SELECT memory_content, embedding FROM user_memory
            WHERE user_id = ?
        ''', (str(user_id),))
        existing_memories = cursor.fetchall()

        if not existing_memories:
            return False  # No existing memories, so not a duplicate

        # Generate embedding for new memory
        new_embedding = embeddings_model.embed_query(new_memory_text)
        new_embedding_array = np.array([new_embedding], dtype='float32')

        # Check similarity with existing memories
        for existing_content, embedding_bytes in existing_memories:
            existing_embedding = np.frombuffer(embedding_bytes, dtype='float32').reshape(1, -1)

            # Calculate L2 distance (lower = more similar)
            distance = np.linalg.norm(new_embedding_array - existing_embedding)
            similarity = 1 / (1 + distance)  # Convert distance to similarity score (0-1)

            if similarity >= similarity_threshold:
                print(f"[DEBUG] Memory duplicate detected: '{new_memory_text}' is similar to '{existing_content}' (similarity: {similarity:.2f})")
                return True

        return False
    except Exception as e:
        print(f"[WARN] Error checking memory duplicate: {e}")
        return False


def extract_user_memories(user_id, latest_message_text, existing_memories_text):
    """
    Extract new user memories from the latest message using LLM with structured output.
    """
    try:
        memory_system_prompt = """You are responsible for updating and maintaining accurate user memory.

TASK:
- Review the user's latest message below
- Extract user-specific info worth storing long-term (identity, preferences, skills, projects, goals)
- For each item: set is_new=true ONLY if it's NEW info not in existing memories
- Keep each memory as a short atomic sentence (max 15 words)
- No speculation; only facts stated by the user
- If nothing memory-worthy exists, set should_write=false"""

        memory_user_message = f"""EXISTING MEMORIES:
{existing_memories_text or "(empty)"}

LATEST USER MESSAGE:
{latest_message_text}

Extract new facts from the latest message. Return structured output."""

        # Create structured output extractor
        memory_extractor = llm.with_structured_output(MemoryDecision)

        # Extract memories with both system and user messages
        decision: MemoryDecision = memory_extractor.invoke([
            SystemMessage(content=memory_system_prompt),
            HumanMessage(content=memory_user_message)
        ])

        print(f"[DEBUG] Memory extraction decision: should_write={decision.should_write}, memories={len(decision.memories)}")
        for mem in decision.memories:
            print(f"[DEBUG]   - '{mem.text}' (is_new={mem.is_new})")

        return decision
    except Exception as e:
        print(f"[ERROR] Error extracting user memories: {e}")
        import traceback
        traceback.print_exc()
        return MemoryDecision(should_write=False, memories=[])


# ======================= RAG Functions =======================

def get_faiss_path(thread_id):
    """Get the FAISS index directory path for a specific thread."""
    return FAISS_BASE_PATH / str(thread_id)

def get_faiss_index_file(thread_id):
    """Get the full path to FAISS index file."""
    return get_faiss_path(thread_id) / "index.faiss"

def get_faiss_metadata_file(thread_id):
    """Get the full path to metadata file."""
    return get_faiss_path(thread_id) / "metadata.pkl"


def process_pdf_for_thread(pdf_path, thread_id, filename):
    """
    Process PDF file: extract text, split into chunks, generate embeddings.
    Store FAISS index and metadata for the thread.
    """
    try:
        # 1. Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            return {"status": "error", "message": "No text found in PDF"}

        # 2. Split into chunks with overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            return {"status": "error", "message": "Could not split PDF into chunks"}

        # 3. Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata['filename'] = filename
            chunk.metadata['chunk_id'] = i
            chunk.metadata['thread_id'] = str(thread_id)

        # 4. Generate embeddings and create FAISS index
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = embeddings_model.embed_documents(chunk_texts)

        # Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)

        embeddings_array = np.array(embeddings).astype('float32')
        index.add(embeddings_array)

        # 5. Save FAISS index and metadata
        thread_path = get_faiss_path(thread_id)
        thread_path.mkdir(exist_ok=True)

        faiss.write_index(index, str(get_faiss_index_file(thread_id)))

        # Save metadata
        metadata = {
            'chunks': [chunk.page_content for chunk in chunks],
            'metadata': [chunk.metadata for chunk in chunks],
            'filenames': [filename]
        }

        with open(get_faiss_metadata_file(thread_id), 'wb') as f:
            pickle.dump(metadata, f)

        return {
            "status": "success",
            "message": f"Processed {filename} with {len(chunks)} chunks",
            "chunk_count": len(chunks)
        }

    except Exception as e:
        return {"status": "error", "message": f"Error processing PDF: {str(e)}"}


def load_faiss_for_thread(thread_id):
    """
    Load FAISS index and metadata for a specific thread.
    Returns: tuple (index, metadata) or (None, None) if not found
    """
    try:
        index_path = get_faiss_index_file(thread_id)
        metadata_path = get_faiss_metadata_file(thread_id)

        if not index_path.exists() or not metadata_path.exists():
            return None, None

        index = faiss.read_index(str(index_path))

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        return index, metadata

    except Exception as e:
        print(f"Error loading FAISS for thread {thread_id}: {str(e)}")
        return None, None



# tool for retrieving information from uploaded documents using FAISS search
@tool
def retrieve_from_documents(query: str, thread_id: str = None) -> str:
    """
    Search uploaded documents for relevant information using semantic similarity.
    Use this tool to find information from PDFs uploaded to the conversation.
    """
    global _CURRENT_THREAD_ID

    # Use provided thread_id or fallback to global context
    current_thread = thread_id or _CURRENT_THREAD_ID
    if not current_thread:
        return "Error: No thread context available. Please start a conversation first."

    try:
        # Load FAISS index for this thread
        index, metadata = load_faiss_for_thread(current_thread)

        if index is None or metadata is None:
            return "No documents uploaded for this conversation yet. Please upload a PDF first."

        # Generate embedding for the query
        query_embedding = embeddings_model.embed_query(query)

        # Search similar chunks
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query_array, TOP_K_CHUNKS)

        if len(indices[0]) == 0:
            return "No relevant documents found for your query."

        # Format results
        results = []
        for idx in indices[0]:
            if idx < len(metadata['chunks']):
                chunk_text = metadata['chunks'][idx]
                chunk_meta = metadata['metadata'][idx]
                filename = chunk_meta.get('filename', 'Unknown')

                results.append(
                    f"**From {filename}:**\n{chunk_text}\n"
                )

        formatted_results = "\n---\n".join(results)
        return f"Found relevant information:\n\n{formatted_results}"

    except Exception as e:
        return f"Error retrieving documents: {str(e)}"


# ======================= Building workflow ==============================================


# state of the workflow
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ======================= Remember Node (Memory Extraction) =======================

def remember_node(state: ChatState):
    """
    Extract and store new user memories from the latest message.
    This node runs BEFORE chat_node to capture user information.
    """
    try:
        # Get current user_id (persistent across all chats)
        user_id = _CURRENT_USER_ID
        if not user_id:
            print(f"[DEBUG] Remember node: No user_id (_CURRENT_USER_ID not set)")
            return {}

        print(f"[DEBUG] Remember node: Processing for user_id={user_id}")

        # Get existing memories
        existing_memories = get_memory_context(user_id)
        print(f"[DEBUG] Existing memories: {existing_memories if existing_memories else '(empty)'}")

        # Get latest user message (last HumanMessage)
        latest_message_text = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                latest_message_text = msg.content
                break

        if not latest_message_text:
            print(f"[DEBUG] Remember node: No user message found in state")
            return {}

        print(f"[DEBUG] Latest message: {latest_message_text[:100]}...")

        # Extract memories from latest message
        decision = extract_user_memories(user_id, latest_message_text, existing_memories)

        print(f"[DEBUG] Extraction decision: should_write={decision.should_write}, count={len(decision.memories)}")

        # Process extracted memories with deduplication
        if decision.should_write and decision.memories:
            for mem in decision.memories:
                print(f"[DEBUG] Processing memory: '{mem.text}' (is_new={mem.is_new})")

                if mem.is_new and mem.text.strip():
                    # Double-check for duplicates before storing
                    if not is_memory_duplicate(mem.text, user_id):
                        # Store as 'fact' type memory
                        success = add_user_memory(user_id, 'fact', mem.text)
                        if success:
                            print(f"[SUCCESS] ✅ New memory stored: '{mem.text}'")
                        else:
                            print(f"[ERROR] Failed to store memory: '{mem.text}'")
                    else:
                        print(f"[DEBUG] ⏭️  Skipped duplicate memory: '{mem.text}'")
                else:
                    print(f"[DEBUG] ⏭️  Skipped (is_new=False): '{mem.text}'")
        else:
            print(f"[DEBUG] No new memories to store")

    except Exception as e:
        print(f"[ERROR] Remember node failed: {e}")
        import traceback
        traceback.print_exc()

    return {}


def chat_node(state: ChatState):
    messages = state['messages']

    # Apply message trimming and summarization
    trimmed_messages = prepare_messages_with_trimming(messages)

    # Build comprehensive system prompt
    system_content = """You are SurAI, a helpful, knowledgeable, and personable AI assistant.

BEHAVIOR
- Be helpful, honest, and conversational
- Provide clear, structured answers with examples when relevant
- Acknowledge limitations and uncertainties
- Use a friendly but professional tone

TOOLS
- Internet Search: Only for current events, recent news, frequently changing info. Skip for general knowledge, history, definitions, or coding.
- Document Retrieval: For uploaded PDFs — search documents before searching internet.
- Stock Price: For current stock quotes/market data only. Not for analysis or advice.
- Calculator: For mathematical calculations with large or complex numbers.

MEMORY & PERSONALIZATION
If user profile is available:
- Address user by name when appropriate
- Reference their skills, projects, or interests contextually
- Tailor explanation depth to their expertise level
- Build on previous context naturally

RESPONSE STRUCTURE
1. Direct answer
2. Explanation/reasoning if helpful
3. Examples or use cases if relevant
4. Follow-up suggestions

FOLLOW-UP QUESTIONS
End each response with 2-3 relevant follow-up questions that build on the answer and help the user explore deeper. Format: "You might also want to know: • Q1? • Q2?"

TONE
- Technical users: precise, correct terminology, no over-explanation
- General users: clear, use analogies, avoid jargon
- Always adapt to user's communication style  
"""

    # Inject user memories into system message
    try:
        if _CURRENT_USER_ID:
            memory_context = get_memory_context(_CURRENT_USER_ID)
            if memory_context:
                system_content += "\n\n=== YOUR PROFILE ===\nRemember these facts about the user:\n" + memory_context
            else:
                system_content += "\n\n=== YOUR PROFILE ===\n(No stored facts yet - learn about the user as you converse)"
    except Exception as e:
        print(f"[WARN] Could not inject user memories: {e}")

    system_instruction = SystemMessage(content=system_content)
    messages_with_system = [system_instruction] + trimmed_messages
    response = llm.invoke(messages_with_system)
    return {"messages": [response]}

# ======================= Tool Binding =======================
# Bind all tools to LLM and create tool node
tools = [search_internet, calculator, get_stock_price, retrieve_from_documents]
llm = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# ======================= Define Graph =======================
# Now that all tools are bound, define the graph
graph = StateGraph(ChatState)

# defining the nodes and edges
graph.add_node("remember", remember_node)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Workflow: START -> remember -> chat_node -> (conditional) tools -> chat_node -> END
graph.add_edge(START, "remember")
graph.add_edge("remember", "chat_node")
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


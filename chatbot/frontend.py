import streamlit as st
from chatbot.backend import chatbot,llm,get_all_threads,save_chat_title
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        #st.session_state['chat_threads'].append(thread_id)
        st.session_state['chat_threads'][thread_id] = 'chat title'

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])
    # Filter out ToolMessages - keep only HumanMessage and AIMessage
    filtered_messages = [msg for msg in messages if not isinstance(msg, ToolMessage)]
    return filtered_messages

def chat_title(thread_id, query):
    title = llm.invoke(f'Generate a concise title for this conversation: {query}')
    title_text =  title.content[0].get('text','')

    # save to the session state
    st.session_state['chat_threads'][thread_id] = title_text

    # save to the database
    save_chat_title(thread_id, title_text)


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()
    print(f"Chat threads loaded: {st.session_state['chat_threads']}")

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('SurAI')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

if st.session_state['chat_threads']:
  for thread_id,title in reversed(st.session_state['chat_threads'].items()):
    if st.sidebar.button(str(title),key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
                content = msg.content
            else:
                role='assistant'
                content = msg.content
                # Extract text from list format (Gemini streaming format)
                if isinstance(content, list) and len(content) > 0:
                    content = content[0].get('text', '')

            # Only add messages with non-empty content
            if content and (isinstance(content, str) and content.strip() or isinstance(content, list) and len(content) > 0):
                temp_messages.append({'role': role, 'content': content})

        st.session_state['message_history'] = temp_messages
else:
   st.sidebar.info("No conversations yet. Start a new chat!")

# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)    
    
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    should_rerun = False
    # if this is the first message of the conversation, generate a title for the thread    
    if len(st.session_state['message_history']) == 1:
        chat_title(st.session_state['thread_id'], user_input)
        should_rerun = True 

    # get the response from the chatbot and stream in to the UI token by token, while also saving the response in the message history
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content
                    if isinstance(content, list) and len(content) > 0:
                        text = content[0].get('text', '')
                        if text.strip():
                            yield text

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    if should_rerun:
        st.rerun()
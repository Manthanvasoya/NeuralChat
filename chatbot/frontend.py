import streamlit as st
from backend import (
    chatbot, llm, get_all_threads, save_chat_title, delete_chat,
    process_pdf_for_thread, load_faiss_for_thread, retrieve_from_documents,
    _CURRENT_THREAD_ID
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import os
import tempfile
import backend
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

    # Handle different response formats
    if isinstance(title.content, str):
        title_text = title.content
    elif isinstance(title.content, list) and len(title.content) > 0:
        first = title.content[0]
        title_text = first.get('text', '') if isinstance(first, dict) else str(first)
    else:
        title_text = 'Chat'

    # save to the session state
    st.session_state['chat_threads'][thread_id] = title_text

    # save to the database
    save_chat_title(thread_id, title_text)

@st.dialog("Delete Chat")
def confirm_delete_dialog(thread_id):
    title_to_delete = st.session_state['chat_threads'].get(thread_id, 'Chat')
    st.warning(f"Are you sure you want to delete '{title_to_delete}'? This action cannot be undone.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Delete", key="confirm_delete_btn", use_container_width=True):
            if delete_chat(thread_id):
                st.session_state['chat_threads'].pop(thread_id, None)

                # If we deleted the current chat, switch to most recent existing chat
                if st.session_state['thread_id'] == thread_id:
                    remaining_chats = list(st.session_state['chat_threads'].keys())
                    if remaining_chats:
                        # Switch to the most recent chat (last one in the dict)
                        st.session_state['thread_id'] = remaining_chats[-1]
                        messages = load_conversation(remaining_chats[-1])
                        temp_messages = []
                        for msg in messages:
                            if isinstance(msg, HumanMessage):
                                role='user'
                                content = msg.content
                            else:
                                role='assistant'
                                content = msg.content
                                if isinstance(content, list) and len(content) > 0:
                                    content = content[0].get('text', '')
                            if content and (isinstance(content, str) and content.strip() or isinstance(content, list) and len(content) > 0):
                                temp_messages.append({'role': role, 'content': content})
                        st.session_state['message_history'] = temp_messages
                    else:
                        # Only create new chat if no chats exist
                        reset_chat()
                st.rerun()

    with col2:
        if st.button("✕ Cancel", key="cancel_delete_btn", use_container_width=True):
            st.rerun()



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

st.sidebar.divider()
st.sidebar.subheader("📄 Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF for this chat",
    type="pdf",
    key=f"pdf_upload_{st.session_state['thread_id']}"
)

if uploaded_file:
    with st.sidebar.status("Processing PDF...", expanded=True) as status:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        # Process PDF
        result = process_pdf_for_thread(
            tmp_path,
            st.session_state['thread_id'],
            uploaded_file.name
        )

        if result['status'] == 'success':
            status.update(label="✅ PDF processed!", state="complete")
            st.sidebar.success(f"✅ {result['message']}")
        else:
            status.update(label="❌ Error processing PDF", state="error")
            st.sidebar.error(f"❌ {result['message']}")

        # Clean up temp file
        os.remove(tmp_path)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Chat Settings")

# Control for keeping recent messages in long conversations
keep_recent = st.sidebar.slider(
    "Keep recent messages:",
    min_value=5,
    max_value=50,
    value=12,
    step=1,
    help="Only these recent messages are sent to the LLM. Older messages are summarized for context. Adjust for longer/shorter context window."
)

# Update backend setting
from backend import set_keep_recent
set_keep_recent(keep_recent)

st.sidebar.divider()
st.sidebar.header('My Conversations')

if st.session_state['chat_threads']:
  for thread_id,title in reversed(st.session_state['chat_threads'].items()):
    col1, col2 = st.sidebar.columns([4, 1])

    with col1:
      if st.button(str(title), key=str(thread_id), use_container_width=True):
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

    with col2:
      if st.button("🗑️", key=f"delete_{thread_id}", help="Delete chat"):
        confirm_delete_dialog(thread_id)
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
        # Add a placeholder for tool status
        status_placeholder = st.empty()
        message_placeholder = st.empty()

        def stream_with_tool_tracking():
            # Set the global thread_id for tool access
            backend._CURRENT_THREAD_ID = st.session_state['thread_id']

            tool_display = ""
            current_message = ""

            # Stream events to capture tool calls
            for event in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="values"
            ):
                # Check if there are tool calls being made
                if 'messages' in event:
                    messages = event['messages']
                    if messages and hasattr(messages[-1], 'tool_calls'):
                        tool_calls = messages[-1].tool_calls
                        if tool_calls:
                            for tool_call in tool_calls:
                                tool_name = tool_call.get('name', 'Unknown Tool')
                                tool_display = f"🔧 Using tool: **{tool_name}**..."
                                status_placeholder.info(tool_display)

                    # Get the latest message content
                    if messages and isinstance(messages[-1], AIMessage):
                        content = messages[-1].content
                        if isinstance(content, str) and content.strip():
                            current_message = content
                            status_placeholder.empty()
                            message_placeholder.write(current_message)
                        elif isinstance(content, list) and len(content) > 0:
                            text = content[0].get('text', '')
                            if text.strip():
                                current_message = text
                                status_placeholder.empty()
                                message_placeholder.write(current_message)

            status_placeholder.empty()
            return current_message

        ai_message = stream_with_tool_tracking()

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    if should_rerun:
        st.rerun()
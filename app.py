import streamlit as st

st.set_page_config(page_title="PDF QA System")
st.title("PDF QA System")

# reset button to clear session state
if st.sidebar.button("Reset Chat"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# Manage session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type='pdf')

# Create a Chat Interface with streamlit_chat
from streamlit_chat import message   

# Function to handle user input and display response
def handle_userinput(user_question):
    with st.spinner('Generating response...'):
        result = st.session_state.conversation.invoke({"question": user_question})
        
        response = result.content if hasattr(result, 'content') else "Sorry, I couldn't retrieve a proper response."
        
        st.session_state.chat_history.append(f"You: {user_question}")
        st.session_state.chat_history.append(f"Bot: {response}")

    # Layout for displaying input and response
    response_container = st.container()
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages, is_user=True, key=str(i))
            else:
                message(messages, key=str(i))



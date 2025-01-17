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


import os   
from langchain_core.documents import Document   
from langchain_community.document_loaders import PyPDFLoader   
from langchain.text_splitter import RecursiveCharacterTextSplitter   


# Function to process the uploaded PDF file
def process_pdf(pdf_file):
    loaders = PyPDFLoader(pdf_file)
    pages = loaders.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    doc_list = []
    
    for page in pages:
        pg_split = text_splitter.split_text(page.page_content)
        for pg_sub_split in pg_split:
            metadata = {"source": "Uploaded PDF"}
            doc_string = Document(page_content=pg_sub_split, metadata=metadata)
            doc_list.append(doc_string)
    
    return doc_list

# Convert Text into Embeddings for Vector Search
from langchain_community.embeddings import HuggingFaceEmbeddings   

embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')


# Store Document Embeddings in Qdrant
from qdrant_client import QdrantClient   
from langchain_qdrant import QdrantVectorStore   

if uploaded_file:
    with open("uploaded_temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    doc_list = process_pdf("uploaded_temp.pdf")
    
    vectorstore = QdrantVectorStore.from_documents(
        doc_list,
        embed_model,
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
        collection_name="RAG_chatbot",
        prefer_grpc=True,
        force_recreate=True
    )
    st.session_state.processComplete = True


# Generate Answers with Google Gemini Model
from langchain_google_genai import ChatGoogleGenerativeAI   
from langchain_core.prompts import ChatPromptTemplate   
from langchain_core.runnables import RunnableSequence, RunnableMap   
from operator import itemgetter   

# retrieval and answer generation pipeline 
def get_qa_chain(vectorstore, num_chunks):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
    
    # prompt template
    prompt_str = """
    Answer the user question based only on the following context:
    {context}

    Question: {question}
    """
    _prompt = ChatPromptTemplate.from_template(prompt_str)
    
    # Initialize the Gemini Chat model
    chat_llm = ChatGoogleGenerativeAI(
        api_key=st.secrets["GEMINI_API_KEY"],
        model="gemini-1.5-flash",
        temperature=0
    )
    
    query_fetcher = itemgetter("question")
    retrieval_pipeline = query_fetcher | retriever
    
    setup_pipeline = RunnableMap({"question": query_fetcher, "context": retrieval_pipeline})
    
    qa_chain = setup_pipeline | _prompt | chat_llm
    
    return qa_chain

# Handle User Input and Display Responses
if st.session_state.processComplete:
    num_chunks = 3
    st.session_state.conversation = get_qa_chain(vectorstore, num_chunks)
    
    user_question = st.chat_input("Ask a question about the PDF:")
    if user_question:
        handle_userinput(user_question)

# Build the Main App
def main():
    st.sidebar.button("Upload PDF and ask questions!")

if __name__ == "__main__":
    main()
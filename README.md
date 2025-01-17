# RAG_Chatbot
---
## Problem Statement:
**Building a Simple Q&A Chatbot with RAG and LangChain.** <br> 
Create a question-answering system that can understand, and answer questions based on PDF documents (like course materials or company documentation). <br>
The system should use LangChain and RAG (RetrievalAugmented Generation) to provide accurate answers based on the document content. <br>
Project Goals

1. Build a working prototype that can:
  A. Read and process PDF documents <br>
  B. Answer questions based on the document content <br>
  C. Provide relevant responses using RAG architecture <br>
2. Learn key concepts:
   A. Document processing with LangChain <br>
   B. Vector embeddings <br>
   C. RAG architecture <br>
   D. LLM integration <br>
---
# Prerequisites:
- Python 3.9+ installed <br>
## Installation:
To install dependancies:
```
pip install -r requirements.txt
```

To run:
```
streamlit run app.py
```
# Notes:
Update ```secrets.toml``` file in ```.streamlit```
```
QDRANT_API_KEY = YOUR API KEY
QDRANT_URL=YOUR API KEY
GEMINI_API_KEY = YOUR API KEY
```

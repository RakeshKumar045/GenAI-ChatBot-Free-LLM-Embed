from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

print("my qan a start")
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


print(api_key)

#cell 1
print("embeddig ")

embHuggingface = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("loading vecrordb")
#cell 2
vectorstroemore = FAISS.load_local(
    "vectormoretxtfastapi",
    embHuggingface,
    allow_dangerous_deserialization=True
)

# cell 3

from langchain_groq import ChatGroq
import os


print("create llm")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    
    temperature=0
)

# cell4

print("write a query")

query = "how i can  connect with mperito and what is the website link?"

print("searching in vector database")

documents = vectorstroemore.similarity_search(
    query,
    k=3
) 

context = "\n\n".join(
    doc.page_content for doc in documents
)

print("adding the prompt")

prompt = f"""
You are an AI assistant for Mperito.

Answer ONLY from the provided context.

Rules:
1. Give only the required information.
2. Answer in bullet points.
3. Each bullet should be one short line.
4. Do not explain.
5. Do not add information that is not in the context.
6. If the answer is not available, reply:
   Information not available in the document.
7. Give the website link in the answer if it is available in the context.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("llm generating answer")

print(response.content)
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

from langchain_community.document_loaders import DirectoryLoader, TextLoader

print("Starting the ingestion process... and loading the documents from the specified directory.")

loader = DirectoryLoader(r"C:\Users\nikky\OneDrive\Documents\internship\chatbotdemo\nikmoraboutmpertio",
    glob="*.txt",
    loader_cls=TextLoader
)


print("Documents loaded successfully from the specified directory.")
#cell 2
documents = loader.load()

print("Total files loaded:", len(documents))
print(documents)

print("Starting the text splitting process...")
#cell 3
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

print(api_key)


print("Text splitting process completed successfully.")
# cell4

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100
)

print("Text splitting process completed successfully.")
# cell5
chunks = text_splitter.split_documents(documents)
print(len(chunks))
chunks[0]

# cell6

print("Starting the embedding process using HuggingFaceEmbeddings...")

embHuggingface = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding process completed successfully.")
# cell7
vectorstoremoretxtdata = FAISS.from_documents(

    documents=chunks,

    embedding=embHuggingface
    
)

print("Vectorstore created successfully from the documents and embeddings.")

# cell8
vectorstoremoretxtdata.save_local("vectormoretxtfastapi")

print("Vectorstore saved locally as 'vectormoretxtfastapi'")

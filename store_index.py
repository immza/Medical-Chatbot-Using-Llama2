# store_index.py

from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY is None:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Load PDF data and split into chunks
extracted_data = load_pdf("data/")  # change path if needed
text_chunks = text_split(extracted_data)
print("Number of text chunks:", len(text_chunks))

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment="us-east-1-aws"  # explicitly set your Pinecone environment
)
)

index_name = "medical-chatbot"

# Check if index exists; create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,        # embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(index_name)

# Create vector store from text chunks
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Or directly from texts (optional, will overwrite the above if used)
vector_store = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name
)

print("Vector store ready and indexed!")

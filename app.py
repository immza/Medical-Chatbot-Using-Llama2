# app.py

from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import CTransformers  # Updated import
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import prompt_template
from pinecone import Pinecone, ServerlessSpec
import os

app = Flask(__name__)
load_dotenv()

# -------------------------
# Environment Variables
# -------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# -------------------------
# Load Embeddings
# -------------------------
embeddings = download_hugging_face_embeddings()

# -------------------------
# Initialize Pinecone Client
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1-aws")
index_name = "medical-chatbot"

# Check if index exists, create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(index_name)

# -------------------------
# Vector Store
# -------------------------
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# -------------------------
# Prompt + LLM Setup
# -------------------------
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={"max_new_tokens": 512, "temperature": 0.8}
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# -------------------------
# Flask Routes
# -------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("msg", "")
    print(f"User Input: {msg}")
    result = qa({"query": msg})
    print("Response: ", result["result"])
    return jsonify({"response": result["result"]})


# -------------------------
# Run Flask App
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

import time
from flask import Flask, render_template, request, jsonify, jsonify
from src.helpers import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from src.prompt import *
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


app = Flask(__name__)


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "medicalchatbotnew"

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)


chain_type = {"prompt": PROMPT}
llm = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model="gemini-2.5-pro")

1
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone initialized successfully")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    exit(1)

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
print(f"Existing indexes: {existing_indexes}")


if INDEX_NAME in existing_indexes:
        print(f"Index {INDEX_NAME} already exists")
else:
        print(f"Creating serverless index: {INDEX_NAME}")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            print("Waiting for index to be ready...")
            while True:
                try:
                    desc = pc.describe_index(INDEX_NAME)
                    if desc.status['ready']:
                        print("Index is ready!")
                        break
                    time.sleep(2)
                except Exception as e:
                    print(f"Waiting... {e}")
                    time.sleep(2)
                    
        except Exception as e:
            print(f"Error creating index: {e}")
            if "already exists" in str(e).lower():
                print("Index already exists, continuing...")
            else:
                exit(1)
                
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)  

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type= "stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == "__main__":
    app.run(debug=True)
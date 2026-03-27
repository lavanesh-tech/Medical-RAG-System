import os

from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

try:
    from langchain.chains.retrieval import create_retrieval_chain
except ImportError:
    from langchain.chains import create_retrieval_chain

from src.helper import download_embeddings
from src.prompt import system_prompt

load_dotenv(override=True)

app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing. Set it in your .env file.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Set it in your .env file.")

embeddings = download_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore(
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
    index_name=index_name,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.4,
    api_key=OPENAI_API_KEY,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def _get_user_message() -> str:
    data = request.get_json(silent=True)
    if isinstance(data, dict) and data.get("msg") is not None:
        return str(data["msg"]).strip()
    return (request.form.get("msg") or request.args.get("msg") or "").strip()


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = _get_user_message()
    if not msg:
        return "Please provide a message in the 'msg' field.", 400

    print(f"User Input: {msg}")

    response = rag_chain.invoke({"input": msg})
    print(f"Response: {response['answer']}")

    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

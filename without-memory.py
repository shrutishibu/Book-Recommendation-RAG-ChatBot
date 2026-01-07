import os
import httpx
import pandas as pd

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")

client = httpx.Client(verify=False)

CSV_PATH = "data.csv"
FAISS_DIR = "faiss_index"

def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="PLACEHOLDER_EMBEDDING_MODEL",
        base_url="PLACEHOLDER_EMBEDDING_ENDPOINT",
        api_key=API_KEY,
        http_client=client
    )

    if os.path.exists(FAISS_DIR):
        print("Loading FAISS index...")
        return FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("Reading CSV...")
    documents = pd.read_csv(CSV_PATH, encoding="utf-8")

    print(f"Total rows in CSV: {len(documents)}")

    # Combine rows into documents
    docs = documents.astype(str).apply(
        lambda row: " | ".join(row.values),
        axis=1
    ).tolist()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_text("\n\n".join(docs))

    print(f"Total chunks created: {len(chunks)}")

    print("Creating FAISS index...")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local(FAISS_DIR)

    print("FAISS index created")
    
    return vectorstore

def answer_question(user_query: str):
    vectorstore = get_vectorstore()

    docs = vectorstore.max_marginal_relevance_search(
        user_query, 
        k=5,
        fetch_k=15)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(
        model="PLACEHOLDER_EMBEDDING_MODEL",
        base_url="PLACEHOLDER_EMBEDDING_ENDPOINT",
        api_key=API_KEY,
        http_client=client,
        temperature=0.1
    )

    answer_prompt = ChatPromptTemplate.from_template(
        """
Answer using ONLY the context below.
If not found, say "answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    )

    response = (
        answer_prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "context": context,
        "question": user_query
    })

    print("\n" + "=" * 70)
    print("FINAL ANSWER:\n")
    print(response)
    print("=" * 70)

if __name__ == "__main__":
    print("\nCSV RAG Chatbot with Memory (type 'exit' to quit)\n")

    while True:
        q = input("Question: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        if q:
            answer_question(q)

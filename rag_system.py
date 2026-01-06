import os
import ssl
import httpx
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ------------------------------------------------------------------
# SSL + ENV SETUP
# ------------------------------------------------------------------
ssl._create_default_https_context = ssl._create_unverified_context
load_dotenv()

API_KEY = os.getenv("GENAILAB_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ GENAILAB_API_KEY not found in .env")

client = httpx.Client(verify=False)

FAISS_DIR = "faiss_index"
CSV_PATH = "data.csv"

# ------------------------------------------------------------------
# BUILD OR LOAD FAISS INDEX FROM CSV
# ------------------------------------------------------------------
def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="azure/genailab-maas-text-embedding-3-large",
        base_url="https://genailab.tcs.in",
        api_key=API_KEY,
        http_client=client
    )

    if os.path.exists(FAISS_DIR):
        print("📦 Loading existing FAISS index...")
        return FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("📄 Reading CSV...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8", errors="ignore")

    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    texts = df.astype(str).apply(
        lambda row: " | ".join(row.values),
        axis=1
    ).tolist()

    print("🔢 Creating embeddings and FAISS index...")
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    vectorstore.save_local(FAISS_DIR)

    print("✅ FAISS index created and saved")
    return vectorstore

# ------------------------------------------------------------------
# CONVERSATIONAL RAG CHAIN
# ------------------------------------------------------------------
def get_conversational_chain():
    llm = ChatOpenAI(
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        base_url="https://genailab.tcs.in",
        api_key=API_KEY,
        http_client=client,
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an assistant answering questions ONLY using the provided context.
If the answer is not present, say:
"answer is not available in the context"

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ------------------------------------------------------------------
# QUERY FUNCTION
# ------------------------------------------------------------------
def answer_question(question: str):
    vectorstore = get_vectorstore()

    print("\n🔍 Running similarity search...")
    docs = vectorstore.similarity_search(question, k=3)

    print("\n📚 Top retrieved chunks:")
    for i, d in enumerate(docs, 1):
        print(f"\n--- DOC {i} ---")
        print(d.page_content[:500])

    context = "\n\n".join(doc.page_content for doc in docs)

    chain = get_conversational_chain()

    print("\n🤖 Generating final answer...\n")
    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER:\n")
    print(response)
    print("=" * 70)

# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n📊 CSV RAG Chatbot (type 'exit' to quit)\n")

    while True:
        user_q = input("❓ Question: ").strip()
        if user_q.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        if user_q:
            answer_question(user_q)

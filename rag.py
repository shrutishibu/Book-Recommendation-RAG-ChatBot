import os
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found")

os.environ["OPENAI_API_KEY"] = api_key

loader = CSVLoader(file_path="data.csv", encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} rows from CSV")

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

print("\nBooks RAG Chatbot (type 'exit' to quit)\n")

while True:
    query = input("❓ Question: ")

    if query.lower() in ["exit", "quit"]:
        print("Bye! Feel free to reach out if you have more questions.")
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an assistant answering questions using ONLY the context below.

Context:
{context}

Question:
{query}

If the answer is not present in the context, say "Not found in data".
"""

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)
    print("-" * 50)

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()  # reads GOOGLE_API_KEY from the .env file

DB_DIR = "chroma_db_gemini"
COLLECTION = "rag_docs"

# must be the SAME embedding settings used in create_database.py
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768,
)

vectorstore = Chroma(
    collection_name=COLLECTION,
    persist_directory=DB_DIR,
    embedding_function=embedding_model
)

if not vectorstore.get(limit=1)["ids"]:
    raise SystemExit("Database is empty. Run:  python create_database.py  first.")

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)


def to_text(content):
    """Gemini may return a string or a list of parts - always give back plain text."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break
    if not query.strip():
        continue

    try:
        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        print(f"\n AI: {to_text(response.content)}")

        pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
        print(f"\n Sources: pages {pages}\n")

    except Exception as e:
        print(f"\n Error: {e}\n (If this says 429 / quota, wait a minute and try again.)\n")

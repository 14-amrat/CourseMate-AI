import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # reads GOOGLE_API_KEY from the .env file

DB_DIR = "chroma_db_gemini"
COLLECTION = "rag_docs"
BATCH_SIZE = 30
PAUSE_SECONDS = 12


def get_embeddings():
    # same settings must be used for indexing and searching
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=768)


def to_text(content):
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def add_in_batches(vectorstore, chunks, ids, progress):
    """Embed chunks in small batches (Gemini free tier has per-minute limits). Skips chunks already stored."""
    already = set(vectorstore.get()["ids"])
    todo = [i for i in range(len(chunks)) if ids[i] not in already]

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start : start + BATCH_SIZE]
        for attempt in range(6):
            try:
                vectorstore.add_documents([chunks[i] for i in batch], ids=[ids[i] for i in batch])
                break
            except Exception as e:
                msg = str(e)
                if "PerDay" in msg:
                    raise RuntimeError("Daily free quota reached. Progress is saved - upload again tomorrow to continue.")
                if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                    time.sleep(20 * (attempt + 1))
                else:
                    raise
        else:
            raise RuntimeError("Still rate limited. Progress is saved - wait a few minutes and click the button again.")

        progress.progress(min(start + BATCH_SIZE, len(todo)) / len(todo),
                          text=f"Embedded {min(start + BATCH_SIZE, len(todo))}/{len(todo)} chunks")
        time.sleep(PAUSE_SECONDS)


st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    st.error("GOOGLE_API_KEY not found. Add it to the .env file and restart the app.")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")


if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        try:
            with st.spinner("Reading and splitting PDF..."):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = uploaded_file.name

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
                ids = [f"{uploaded_file.name}-{i}" for i in range(len(chunks))]

            vectorstore = Chroma(
                collection_name=COLLECTION,
                embedding_function=get_embeddings(),
                persist_directory=DB_DIR
            )

            st.info(f"{len(chunks)} chunks to index. Large books take several minutes on the free tier.")
            bar = st.progress(0.0, text="Starting...")
            add_in_batches(vectorstore, chunks, ids, bar)
            st.success("Vector database created!")

        except Exception as e:
            st.error(str(e))


if os.path.exists(DB_DIR):

    vectorstore = Chroma(
        collection_name=COLLECTION,
        persist_directory=DB_DIR,
        embedding_function=get_embeddings()
    )

    if vectorstore.get(limit=1)["ids"]:

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

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

        st.divider()
        st.subheader("Ask Questions From the Book")

        query = st.text_input("Enter your question")

        if query:
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

                st.write("### AI Answer")
                st.write(to_text(response.content))

                pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
                st.caption(f"Sources: pages {pages}")

            except Exception as e:
                st.error(f"{e}\n\nIf this mentions 429 / quota, wait a minute and try again.")

# load pdf
# split into chunks
# create the embeddings (Gemini)
# store into chroma
import sys
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # reads GOOGLE_API_KEY from the .env file

PDF_PATH = "document loaders/deeplearning.pdf"
DB_DIR = "chroma_db_gemini"     # NEW folder (old chroma_db was built with OpenAI vectors - not compatible)
COLLECTION = "rag_docs"

# Gemini free tier has per-minute limits, so we embed in small batches with a pause.
BATCH_SIZE = 30
PAUSE_SECONDS = 12
LIMIT = None   # e.g. set LIMIT = 100 to index only 100 chunks for a quick test

docs = PyPDFLoader(PDF_PATH).load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
ids = [f"deeplearning-{i}" for i in range(len(chunks))]
print(f"PDF pages: {len(docs)} | chunks: {len(chunks)}")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768,
)

vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embedding_model,
    persist_directory=DB_DIR,
)

# Resume support: skip chunks that are already stored
already = set(vectorstore.get()["ids"])
todo = [i for i in range(len(chunks)) if ids[i] not in already]
if LIMIT:
    todo = todo[:LIMIT]
print(f"Already stored: {len(already)} | to embed now: {len(todo)}")

for start in range(0, len(todo), BATCH_SIZE):
    batch = todo[start : start + BATCH_SIZE]

    for attempt in range(6):
        try:
            vectorstore.add_documents(
                [chunks[i] for i in batch],
                ids=[ids[i] for i in batch],
            )
            break
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg:
                print("\nDaily free quota reached. Progress is saved - run this script again tomorrow.")
                sys.exit(0)
            if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                wait = 20 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s ...")
                time.sleep(wait)
            else:
                raise
    else:
        print("Still rate limited. Progress is saved - wait a few minutes and run again.")
        sys.exit(1)

    print(f"Embedded {min(start + BATCH_SIZE, len(todo))}/{len(todo)} chunks")
    time.sleep(PAUSE_SECONDS)

print(f"\nDone. Vector database saved in ./{DB_DIR}")

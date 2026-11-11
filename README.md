# 📚 RAG Book Assistant — Gemini + LangChain + ChromaDB

A complete **Retrieval-Augmented Generation (RAG)** project built using **Google Gemini, LangChain, Gemini Embeddings, ChromaDB, and Streamlit**.

This project demonstrates how to build a question-answering system that can read information from documents, convert the document content into vector embeddings, store those embeddings in a vector database, retrieve the most relevant pieces of information for a user's question, and finally use Google Gemini to generate an answer based only on the retrieved context.

The project contains both:

* 🖥️ A **terminal-based RAG application**
* 🌐 A **Streamlit web-based RAG application**

---

# 🚀 Project Overview

Traditional LLM applications have an important limitation:

> An LLM can only reliably answer using the knowledge available to it or the information provided in the current prompt.

RAG solves this problem by giving the LLM relevant information from an external knowledge source before generating the answer.

This project follows the following pipeline:

```text
                    ┌──────────────────────┐
                    │      PDF Document    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    PyPDFLoader       │
                    │   Load PDF pages     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Text Splitter        │
                    │ Chunk size = 1000    │
                    │ Overlap = 200        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Gemini Embeddings    │
                    │ gemini-embedding-001 │
                    │ Dimension = 768      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     ChromaDB         │
                    │   Vector Database    │
                    └──────────┬───────────┘
                               │
                       User asks question
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Query Embedding      │
                    │ Gemini Embeddings    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ MMR Retriever        │
                    │ k = 4                │
                    │ fetch_k = 10         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Relevant Context     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Gemini 2.5 Flash     │
                    │ Answer Generation    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Final Answer     │
                    │ + Source Pages       │
                    └──────────────────────┘
```

---

# ✨ Features

## Core RAG Features

* 📄 PDF document loading
* ✂️ Recursive text splitting
* 🧠 Gemini text embeddings
* 🗃️ ChromaDB vector storage
* 🔎 Semantic similarity search
* 🔀 Maximum Marginal Relevance (MMR) retrieval
* 🤖 Gemini 2.5 Flash answer generation
* 📚 Source page display
* 💾 Persistent vector database
* ♻️ Resume support while creating embeddings
* ⏱️ Rate-limit handling
* 🌐 Streamlit user interface
* 💻 Terminal-based question-answering interface

---

# 🧠 Technologies Used

| Technology       | Purpose                         |
| ---------------- | ------------------------------- |
| Python           | Main programming language       |
| LangChain        | RAG application framework       |
| Google Gemini    | LLM + embedding model           |
| Gemini 2.5 Flash | Answer generation               |
| Gemini Embedding | Document/query embeddings       |
| ChromaDB         | Vector database                 |
| PyPDF            | PDF processing                  |
| Streamlit        | Web interface                   |
| python-dotenv    | Environment variable management |

---

# 🤖 Gemini Models Used

This project uses two Gemini models for different purposes.

## 1. Gemini Embedding Model

```text
gemini-embedding-001
```

Used for converting text into numerical vectors.

Example:

```text
"Deep learning is a subset of machine learning."
```

becomes conceptually:

```text
[0.021, -0.183, 0.421, ...]
```

These vectors allow ChromaDB to perform semantic retrieval.

The project uses:

```python
GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)
```

### Important

The same embedding model and dimensionality must be used during:

1. Document indexing
2. Query retrieval

Therefore both use:

```text
gemini-embedding-001
768 dimensions
```

Do not create the database with one embedding model and query it using a different embedding model.

---

# 2. Gemini 2.5 Flash

The project uses:

```text
gemini-2.5-flash
```

for generating the final answer.

The LLM receives:

```text
Retrieved Context
+
User Question
```

and generates the final response.

---

# 🏗️ Project Structure

```text
GenAI-part2-Rag-implementation/
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
│
├── app.py
├── create_database.py
├── main.py
│
├── document loaders/
│   ├── GRU.pdf
│   ├── deeplearning.pdf
│   ├── notes.txt
│   ├── page.py
│   ├── pdf.py
│   └── test.py
│
├── retrievers/
│   ├── arixv.py
│   ├── mmr.py
│   └── multiquery.py
│
└── vector store/
    └── DB.py
```

After running the database creation script, another directory will be generated:

```text
chroma_db_gemini/
```

This contains the persistent ChromaDB data.

---

# 📂 Important Files

## `create_database.py`

This is the main **document ingestion and indexing pipeline**.

It performs:

```text
PDF
 ↓
Load
 ↓
Split
 ↓
Embedding
 ↓
ChromaDB
```

Run:

```bash
python create_database.py
```

This creates:

```text
chroma_db_gemini/
```

---

# 📂 `main.py`

This is the **terminal-based RAG application**.

It performs:

```text
User Question
      ↓
Embedding
      ↓
ChromaDB
      ↓
MMR Retrieval
      ↓
Context
      ↓
Gemini
      ↓
Answer
```

Run:

```bash
python main.py
```

---

# 📂 `app.py`

This is the **Streamlit web application**.

It allows users to:

1. Upload a PDF
2. Create a vector database
3. Ask questions
4. Retrieve relevant document chunks
5. Generate answers using Gemini
6. See source page numbers

Run:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 📂 `retrievers/mmr.py`

This file demonstrates the difference between:

```text
Similarity Search
```

and:

```text
MMR Search
```

It uses small example documents related to machine learning.

---

# 📂 `retrievers/multiquery.py`

This demonstrates LangChain's:

```text
MultiQueryRetriever
```

Instead of relying on only one interpretation of a user's question, MultiQueryRetriever can generate multiple query variations and retrieve information using them.

---

# 📂 `retrievers/arixv.py`

This demonstrates:

```text
ArxivRetriever
```

It retrieves research papers from arXiv.

This file is an experimental/example retriever and is not required for the main PDF RAG application.

---

# 📂 `vector store/DB.py`

This is a small ChromaDB demonstration.

It creates documents such as:

```text
Python is widely used in Artificial Intelligence.

Pandas is used for data analysis in Python.

Neural networks are used in deep learning.
```

and stores them in:

```text
chroma_db_demo/
```

It then performs similarity search.

---

# 📂 `document loaders/`

This directory contains examples of different LangChain document loaders.

## `pdf.py`

Demonstrates:

```python
PyPDFLoader
```

for loading PDF files.

---

## `page.py`

Demonstrates:

```python
WebBaseLoader
```

for loading content from a web page.

---

## `test.py`

Demonstrates:

```python
TextLoader
```

for loading a text file.

---

# 🔑 Environment Setup

The project requires a Google Gemini API key.

Create a file:

```text
.env
```

The project provides:

```text
.env.example
```

with:

```env
GOOGLE_API_KEY=paste_your_key_here
```

Replace the value with your actual API key:

```env
GOOGLE_API_KEY=YOUR_ACTUAL_API_KEY
```

Never commit `.env` to GitHub.

---

# 🔐 Getting a Gemini API Key

Create an API key using Google AI Studio.

Open:

https://aistudio.google.com/apikey

Create your API key and place it in:

```text
.env
```

Example:

```env
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXX
```

Do not share your API key publicly.

---

# 🖥️ Installation — Windows

The following commands are recommended for Windows PowerShell.

## Step 1 — Clone or extract the project

Move into the project directory:

```powershell
cd "C:\path\to\GenAI-part2-Rag-implementation"
```

For example:

```powershell
cd "$HOME\Desktop\GenAI-part2-Rag-implementation"
```

Check the files:

```powershell
dir
```

---

# Step 2 — Create Virtual Environment

```powershell
python -m venv venv
```

This creates:

```text
venv/
```

A virtual environment keeps the project's Python packages separate from your system Python installation.

---

# Step 3 — Activate Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

You should see:

```text
(venv)
```

at the beginning of your terminal.

Example:

```text
(venv) PS C:\Users\User\Desktop\GenAI-part2-Rag-implementation>
```

---

# Step 4 — Upgrade pip

```powershell
python -m pip install --upgrade pip
```

---

# Step 5 — Install Dependencies

```powershell
pip install -r requirements.txt
```

The important packages include:

```text
langchain
langchain-core
langchain-community
langchain-text-splitters
langchain-google-genai
langchain-chroma
chromadb
pypdf
python-dotenv
streamlit
```

---

# Step 6 — Create `.env`

Copy:

```text
.env.example
```

to:

```text
.env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

Open it:

```powershell
notepad .env
```

Add:

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

Save the file.

---

# Step 7 — Verify API Key

Run:

```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API key loaded:', bool(os.getenv('GOOGLE_API_KEY')))"
```

Expected:

```text
API key loaded: True
```

If you get:

```text
API key loaded: False
```

check your `.env` file.

---

# 🗃️ Creating the Vector Database

Before asking questions, documents must be converted into embeddings and stored in ChromaDB.

Run:

```powershell
python create_database.py
```

The script uses:

```text
document loaders/deeplearning.pdf
```

as the source document.

---

# 🔄 What Happens During Database Creation?

## Step 1 — Load PDF

```python
docs = PyPDFLoader(PDF_PATH).load()
```

The PDF is loaded page by page.

---

## Step 2 — Split Documents

The project uses:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

This converts large documents into smaller chunks.

Example:

```text
Original PDF
     ↓
Large document
     ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
...
```

### Chunk Size

```text
1000 characters
```

approximately defines the maximum size of each chunk.

### Chunk Overlap

```text
200 characters
```

keeps some content from the previous chunk.

This helps preserve context across chunk boundaries.

---

# 🧠 Step 3 — Generate Embeddings

Each chunk is passed to:

```text
gemini-embedding-001
```

The text becomes a numerical vector.

Conceptually:

```text
Text
 ↓
Embedding Model
 ↓
[0.12, -0.41, 0.72, ...]
```

---

# 🗃️ Step 4 — Store in ChromaDB

The vectors are stored in:

```text
chroma_db_gemini/
```

The collection is:

```text
rag_docs
```

The database can then be searched using semantic similarity.

---

# ♻️ Resume Support

The database creation script checks which document chunks already exist.

```python
already = set(vectorstore.get()["ids"])
```

Then:

```python
todo = [i for i in range(len(chunks)) if ids[i] not in already]
```

This means that if the process stops halfway through, running:

```bash
python create_database.py
```

again does not necessarily re-embed already stored chunks.

This is particularly useful because embedding APIs may have rate limits.

---

# ⏱️ Batch Processing

The project uses:

```python
BATCH_SIZE = 30
```

and:

```python
PAUSE_SECONDS = 12
```

The reason is to avoid sending too many embedding requests too quickly.

The code also retries temporary errors such as:

```text
429
RESOURCE_EXHAUSTED
503
UNAVAILABLE
```

---

# 💬 Running the Terminal RAG Application

After creating the vector database:

```powershell
python main.py
```

You should see:

```text
Rag system created
press 0 to exit
You :
```

Enter a question:

```text
You : What is deep learning?
```

The application:

```text
Question
   ↓
Query Embedding
   ↓
ChromaDB
   ↓
MMR Retriever
   ↓
Relevant Chunks
   ↓
Prompt
   ↓
Gemini
   ↓
Answer
```

Example output:

```text
AI: Deep learning is ...

Sources: pages [1, 3]
```

To exit:

```text
0
```

---

# 🌐 Running the Streamlit Application

The Streamlit interface provides a graphical way to use the RAG system.

Run:

```powershell
streamlit run app.py
```

Streamlit will display something similar to:

```text
Local URL: http://localhost:8501
```

Open:

```text
http://localhost:8501
```

---

# 📤 Uploading a PDF

The Streamlit application allows you to upload a PDF:

```text
Upload a PDF book
```

After selecting a PDF:

```text
PDF uploaded successfully!
```

Click:

```text
Create Vector Database
```

The application then:

```text
PDF
 ↓
PyPDFLoader
 ↓
Text Splitting
 ↓
Gemini Embeddings
 ↓
ChromaDB
```

---

# 🔎 Asking Questions

After the vector database exists:

```text
Ask Questions From the Book
```

Enter a question.

For example:

```text
What is deep learning?
```

The application retrieves relevant chunks and sends them to Gemini.

---

# 🎯 MMR Retrieval

The project uses:

```python
search_type="mmr"
```

MMR means:

> Maximum Marginal Relevance

The goal is not simply to retrieve the most similar documents.

It tries to balance:

```text
Relevance
+
Diversity
```

This reduces the chance of retrieving several nearly identical chunks.

---

# ⚙️ MMR Configuration

The project uses:

```python
search_kwargs={
    "k": 4,
    "fetch_k": 10,
    "lambda_mult": 0.5
}
```

## `k`

```text
k = 4
```

The final number of documents returned to the LLM is 4.

---

## `fetch_k`

```text
fetch_k = 10
```

The retriever first considers 10 candidate documents.

From those candidates, it selects the final 4.

---

## `lambda_mult`

```text
lambda_mult = 0.5
```

This balances:

```text
Relevance
```

and:

```text
Diversity
```

A higher value gives more importance to relevance.

A lower value gives more importance to diversity.

---

# 🧩 Prompt Design

The system prompt is intentionally strict:

```text
Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:
"I could not find the answer in the document."
```

This is an important RAG technique.

Instead of asking Gemini to answer from its general knowledge, the application instructs it to rely on the retrieved document context.

---

# 🔍 Complete Query Flow

Suppose the user asks:

```text
What is gradient descent?
```

## Step 1

The question is converted into an embedding:

```text
Question
 ↓
Gemini Embedding
 ↓
Query Vector
```

---

## Step 2

The query vector is compared with vectors stored in ChromaDB.

```text
Query Vector
       ↓
   ChromaDB
       ↓
Candidate Chunks
```

---

## Step 3

MMR selects useful and diverse chunks.

```text
Candidate Chunks
       ↓
      MMR
       ↓
Top 4 chunks
```

---

## Step 4

The chunks become context:

```text
Context:
Chunk 1
Chunk 2
Chunk 3
Chunk 4
```

---

## Step 5

The prompt is constructed:

```text
Context:
...

Question:
What is gradient descent?
```

---

## Step 6

Gemini 2.5 Flash generates the answer.

```text
Gemini 2.5 Flash
       ↓
Final Answer
```

---

# 📚 Source Tracking

The application also displays source page numbers.

It uses:

```python
pages = sorted({
    d.metadata.get("page", 0) + 1
    for d in docs
})
```

The `+1` is used because PDF page metadata is commonly zero-indexed.

Therefore:

```text
page = 0
```

is displayed as:

```text
page 1
```

---

# 🧪 Experimental Retriever Examples

The project also contains several learning examples.

---

## Similarity Search

In:

```text
retrievers/mmr.py
```

you can compare:

```text
Similarity Search
```

against:

```text
MMR Search
```

Similarity search primarily focuses on finding the most semantically similar documents.

Example:

```python
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

---

# 🔀 MMR Search

The same file demonstrates:

```python
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)
```

This allows you to experiment with the difference between similarity-based retrieval and diversity-aware retrieval.

---

# 🔎 MultiQueryRetriever

File:

```text
retrievers/multiquery.py
```

MultiQueryRetriever uses an LLM to generate alternative versions of a query.

For example:

```text
Original:
What is gradient descent?
```

Possible conceptual variations:

```text
How does gradient descent work?

What is the purpose of gradient descent?

How is gradient descent used in machine learning?
```

These different formulations can help retrieve relevant information that might not be found using only the original wording.

---

# 📄 Arxiv Retriever

File:

```text
retrievers/arixv.py
```

This demonstrates retrieving academic papers from arXiv.

Example query:

```text
large language models
```

The retriever can return metadata such as:

```text
Title
Authors
Summary
```

This is an experimental component and is not required for the main PDF RAG application.

---

# 🗃️ ChromaDB Demonstration

File:

```text
vector store/DB.py
```

contains a simple example of storing documents in ChromaDB.

Example documents:

```text
Python is widely used in Artificial Intelligence.

Pandas is used for data analysis in Python.

Neural networks are used in deep learning.
```

The script creates:

```text
chroma_db_demo/
```

and performs semantic search.

---

# 📄 Document Loader Examples

The project demonstrates multiple document loaders.

## PDF

```python
PyPDFLoader
```

Used for:

```text
PDF → Documents
```

---

## Text File

```python
TextLoader
```

Used for:

```text
TXT → Documents
```

---

## Web Page

```python
WebBaseLoader
```

Used for:

```text
Web Page → Documents
```

---

# 📦 Requirements

The main dependencies are:

```text
langchain
langchain-core
langchain-community
langchain-classic
langchain-text-splitters

langchain-google-genai

langchain-chroma
chromadb

pypdf
beautifulsoup4
lxml
arxiv

python-dotenv
streamlit
requests
```

Install everything using:

```powershell
pip install -r requirements.txt
```

---

# 🧹 `.gitignore`

The following should not be committed to GitHub:

```gitignore
venv/
.env
__pycache__/
*.pyc
chroma_db_gemini/
.streamlit/
```

The `.env` file contains your API key and must remain private.

---

# ⚠️ Important: Embedding Compatibility

The vector database depends on the embedding model.

This project uses:

```text
gemini-embedding-001
```

with:

```text
768 dimensions
```

The same configuration must be used when querying the database.

Do not create the database with:

```text
OpenAI Embeddings
```

and then query it with:

```text
Gemini Embeddings
```

Likewise, do not arbitrarily change the embedding dimensionality without rebuilding the database.

If the embedding model changes, delete/rebuild the corresponding vector database.

---

# 🗑️ Rebuilding the Vector Database

If you want to completely rebuild the database, stop the application and remove:

```text
chroma_db_gemini/
```

PowerShell:

```powershell
Remove-Item -Recurse -Force .\chroma_db_gemini
```

Then recreate it:

```powershell
python create_database.py
```

---

# 🧪 Quick Test

For a quick test without processing the entire PDF, open:

```text
create_database.py
```

and change:

```python
LIMIT = None
```

to something such as:

```python
LIMIT = 100
```

Then run:

```powershell
python create_database.py
```

This allows you to test the RAG pipeline with a limited number of chunks.

After testing, change it back:

```python
LIMIT = None
```

---

# 🚨 Troubleshooting

## 1. `GOOGLE_API_KEY not found`

Error:

```text
GOOGLE_API_KEY not found
```

Solution:

Create:

```text
.env
```

and add:

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

Then restart the application.

---

# 2. Database is empty

If `main.py` says:

```text
Database is empty.
Run: python create_database.py first.
```

Run:

```powershell
python create_database.py
```

Then:

```powershell
python main.py
```

---

# 3. `429 RESOURCE_EXHAUSTED`

This generally means an API rate/quota limit has been reached.

The project already contains retry logic.

You can:

```text
Wait
↓
Run again
```

For large documents, embedding can take several minutes.

---

# 4. Daily quota reached

The program may display:

```text
Daily free quota reached.
```

If progress was saved, run the indexing script again when the quota becomes available.

```powershell
python create_database.py
```

---

# 5. PowerShell cannot activate virtual environment

If:

```powershell
.\venv\Scripts\Activate.ps1
```

is blocked, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.\venv\Scripts\Activate.ps1
```

---

# 6. `python` command not found

Check:

```powershell
python --version
```

If Python is installed but `python` is unavailable, check:

```powershell
py --version
```

You can then create the environment with:

```powershell
py -m venv venv
```

---

# 7. Streamlit command not found

Instead of:

```powershell
streamlit run app.py
```

use:

```powershell
python -m streamlit run app.py
```

---

# 8. Chroma database problems

If you changed the embedding model or embedding dimension, rebuild the database.

Delete:

```powershell
Remove-Item -Recurse -Force .\chroma_db_gemini
```

Then:

```powershell
python create_database.py
```

---

# 9. Wrong PDF path

The default indexing script expects:

```text
document loaders/deeplearning.pdf
```

Make sure the file exists.

PowerShell:

```powershell
Test-Path ".\document loaders\deeplearning.pdf"
```

Expected:

```text
True
```

---

# 🏃 Complete Windows Run Guide

After extracting the project, execute these commands in order.

## 1. Navigate to project

```powershell
cd "$HOME\Desktop\GenAI-part2-Rag-implementation"
```

## 2. Create virtual environment

```powershell
python -m venv venv
```

## 3. Activate it

```powershell
.\venv\Scripts\Activate.ps1
```

## 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## 5. Install packages

```powershell
pip install -r requirements.txt
```

## 6. Create `.env`

```powershell
Copy-Item .env.example .env
```

## 7. Add API key

```powershell
notepad .env
```

Add:

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

## 8. Verify API key

```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API key loaded:', bool(os.getenv('GOOGLE_API_KEY')))"
```

Expected:

```text
API key loaded: True
```

## 9. Create vector database

```powershell
python create_database.py
```

## 10. Run terminal application

```powershell
python main.py
```

## 11. Run web application

Open another terminal:

```powershell
cd "$HOME\Desktop\GenAI-part2-Rag-implementation"
```

Activate:

```powershell
.\venv\Scripts\Activate.ps1
```

Run:

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# ⚡ Quick Start

If everything is already installed and `.env` is configured:

```powershell
python create_database.py
```

Then:

```powershell
python main.py
```

Or:

```powershell
streamlit run app.py
```

---

# 🔬 RAG Concepts Demonstrated

This project is useful for learning the following concepts:

## 1. Document Loading

```text
PDF
TXT
Web
```

---

## 2. Document Chunking

```text
Large Document
      ↓
Small Chunks
```

---

## 3. Embeddings

```text
Text
 ↓
Vector
```

---

## 4. Vector Databases

```text
Vectors
 ↓
ChromaDB
```

---

## 5. Semantic Search

Search based on meaning rather than only exact keywords.

---

## 6. Similarity Search

Find documents that are closest to the query vector.

---

## 7. MMR

Retrieve relevant information while reducing redundant results.

---

## 8. Multi-Query Retrieval

Generate multiple interpretations of a query to improve retrieval.

---

## 9. Prompt Engineering

Provide retrieved context to the LLM.

---

## 10. Grounded Generation

Ask the LLM to answer using the retrieved document context.

---

# 🧠 What You Should Understand After Completing This Project

After completing this project, you should be able to explain:

```text
What is RAG?
```

```text
Why do we need embeddings?
```

```text
What is a vector database?
```

```text
How does semantic search work?
```

```text
What is ChromaDB?
```

```text
What is chunking?
```

```text
Why do chunks overlap?
```

```text
What is similarity search?
```

```text
What is MMR?
```

```text
What is MultiQueryRetriever?
```

```text
What is the difference between an embedding model
and a generative LLM?
```

```text
How does a question reach the LLM?
```

```text
How does the retrieved context reduce hallucination?
```

---

# 🧩 Complete RAG Architecture

The complete application can be understood as two separate pipelines.

## Pipeline 1 — Indexing

```text
             DOCUMENT
                 │
                 ▼
          Document Loader
                 │
                 ▼
          Text Extraction
                 │
                 ▼
          Text Splitter
                 │
                 ▼
          Document Chunks
                 │
                 ▼
       Gemini Embedding Model
                 │
                 ▼
            Embeddings
                 │
                 ▼
             ChromaDB
```

---

# Pipeline 2 — Question Answering

```text
             USER QUESTION
                   │
                   ▼
          Gemini Embedding
                   │
                   ▼
             Query Vector
                   │
                   ▼
              ChromaDB
                   │
                   ▼
          Candidate Documents
                   │
                   ▼
                 MMR
                   │
                   ▼
          Relevant Documents
                   │
                   ▼
              Context
                   │
                   ▼
             Prompt Template
                   │
                   ▼
          Gemini 2.5 Flash
                   │
                   ▼
             Final Answer
                   │
                   ▼
             Source Pages
```

---

# 🆚 RAG vs Normal LLM

## Normal LLM

```text
User Question
      ↓
     LLM
      ↓
   Answer
```

The model does not automatically search your private PDF.

---

## RAG

```text
User Question
      ↓
Vector Search
      ↓
Relevant Document Chunks
      ↓
Context + Question
      ↓
LLM
      ↓
Grounded Answer
```

This makes it possible to build applications over private or domain-specific documents.

---

# 🔐 Security Notes

Never commit:

```text
.env
```

to GitHub.

Never hard-code:

```python
GOOGLE_API_KEY = "..."
```

inside Python source code.

Use:

```env
GOOGLE_API_KEY=...
```

instead.

The `.gitignore` should contain:

```gitignore
.env
venv/
__pycache__/
*.pyc
chroma_db_gemini/
```

---

# 💰 API Usage

This project sends data to Google's Gemini API for:

1. Generating embeddings
2. Generating answers

Large PDFs can generate many embedding requests.

Therefore:

```text
Small document
     ↓
Fewer chunks
     ↓
Fewer embedding requests
```

while:

```text
Large document
     ↓
Many chunks
     ↓
More embedding requests
```

The project therefore uses batching and pauses between batches.

---

# 📈 Possible Future Improvements

This project can be extended significantly.

## 1. Multiple PDF Support

Allow users to upload multiple documents.

---

## 2. Document Filtering

Filter retrieval by:

```text
filename
document type
page
metadata
```

---

## 3. Better Source Citations

Display:

```text
Document
Page
Chunk
```

alongside every answer.

---

## 4. Conversation Memory

Allow:

```text
User:
What is deep learning?

User:
What are its applications?

User:
Explain the second one.
```

The application would remember previous questions.

---

## 5. Hybrid Search

Combine:

```text
Vector Search
+
Keyword Search
```

---

## 6. Reranking

Use a reranker after initial retrieval:

```text
ChromaDB
   ↓
Top 20
   ↓
Reranker
   ↓
Top 5
   ↓
LLM
```

---

## 7. Metadata Filtering

Store metadata such as:

```json
{
    "source": "deeplearning.pdf",
    "page": 12,
    "document_type": "book"
}
```

and use it during retrieval.

---

## 8. Production Database

For larger applications, alternatives can include:

```text
Qdrant
Pinecone
Weaviate
Milvus
pgvector
```

---

## 9. Authentication

Add:

```text
User Registration
Login
JWT
```

---

## 10. Cloud Deployment

The application can eventually be deployed using:

```text
Docker
+
AWS
+
Cloud deployment
```

---

# 🎯 Learning Roadmap After This Project

A good progression is:

```text
RAG Fundamentals
       ↓
LangChain
       ↓
Embeddings
       ↓
Vector Databases
       ↓
Retrievers
       ↓
MMR
       ↓
MultiQuery
       ↓
Reranking
       ↓
Hybrid Search
       ↓
Conversation RAG
       ↓
Agentic RAG
       ↓
Evaluation
       ↓
Production RAG
       ↓
Cloud Deployment
```

---

# 📝 Example Questions

After indexing the deep learning PDF, try questions such as:

```text
What is deep learning?
```

```text
What is the difference between machine learning and deep learning?
```

```text
What are neural networks?
```

```text
What are the applications of deep learning?
```

```text
What is gradient descent?
```

```text
How are neural networks trained?
```

```text
What are the limitations discussed in the document?
```

The answer should be based on the retrieved document context.

---

# 🛠️ Development Commands

## Install dependencies

```powershell
pip install -r requirements.txt
```

## Create database

```powershell
python create_database.py
```

## Run CLI

```powershell
python main.py
```

## Run Streamlit

```powershell
streamlit run app.py
```

## Check Python

```powershell
python --version
```

## Check pip

```powershell
pip --version
```

## Check installed packages

```powershell
pip list
```

## Check API key

```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(bool(os.getenv('GOOGLE_API_KEY')))"
```

---

# 🧹 Clean Rebuild

If you want to completely start over:

### Delete the vector database

```powershell
Remove-Item -Recurse -Force .\chroma_db_gemini
```

### Recreate it

```powershell
python create_database.py
```

### Run the application

```powershell
python main.py
```

or:

```powershell
streamlit run app.py
```

---

# 📌 Important Notes

### 1. Create the database before running `main.py`

Correct:

```text
create_database.py
        ↓
chroma_db_gemini
        ↓
main.py
```

Incorrect:

```text
main.py
        ↓
No database
        ↓
Error
```

---

### 2. Keep embedding configuration identical

Use:

```text
Model:
gemini-embedding-001

Dimensions:
768
```

for both indexing and retrieval.

---

### 3. Keep your API key private

Never upload:

```text
.env
```

to GitHub.

---

### 4. Large PDFs may take time

Embedding a large PDF requires many API requests.

Do not assume the application is frozen simply because indexing takes several minutes.

---

# 📊 Project Summary

| Component           | Implementation                 |
| ------------------- | ------------------------------ |
| Language            | Python                         |
| Framework           | LangChain                      |
| LLM                 | Gemini 2.5 Flash               |
| Embedding           | Gemini Embedding               |
| Embedding Model     | gemini-embedding-001           |
| Embedding Dimension | 768                            |
| Vector Database     | ChromaDB                       |
| PDF Loader          | PyPDFLoader                    |
| Text Splitter       | RecursiveCharacterTextSplitter |
| Main Retriever      | MMR                            |
| Web UI              | Streamlit                      |
| Environment         | python-dotenv                  |
| Storage             | Persistent ChromaDB            |

---

# ⭐ Final Architecture

```text
                         RAG BOOK ASSISTANT
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
              ▼                                   ▼
       INDEXING PIPELINE                    QUERY PIPELINE
              │                                   │
              ▼                                   ▼
           PDF FILE                         USER QUESTION
              │                                   │
              ▼                                   ▼
        PyPDFLoader                      Gemini Embedding
              │                                   │
              ▼                                   ▼
        Text Splitter                       Query Vector
              │                                   │
              ▼                                   ▼
      Gemini Embeddings                     ChromaDB
              │                                   │
              ▼                                   ▼
          ChromaDB                              MMR
              │                                   │
              └─────────────────┬─────────────────┘
                                │
                                ▼
                         Retrieved Context
                                │
                                ▼
                       Prompt Construction
                                │
                                ▼
                       Gemini 2.5 Flash
                                │
                                ▼
                           AI Answer
                                │
                                ▼
                         Source Page Numbers
```

---

# 👨‍💻 Author

**Mahendra Yadav**

Computer Science Engineering Student

Interested in:

* Software Development
* Backend Engineering
* Generative AI
* RAG Systems
* Cloud Computing
* Distributed Systems
* Database Systems

---

# ⭐ If You Found This Project Useful

Consider improving the project further by adding:

```text
✅ Multiple document support
✅ Chat history
✅ Hybrid retrieval
✅ Reranking
✅ Better citations
✅ Authentication
✅ Evaluation
✅ Docker
✅ Cloud deployment
```

The current project provides a strong foundation for understanding how a basic **production-oriented RAG pipeline** works from document ingestion to retrieval and LLM generation.

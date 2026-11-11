from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=768)

vectorstore = Chroma.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query)


print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)

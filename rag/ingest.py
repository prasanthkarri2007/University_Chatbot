import os

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document

from rag.semantic_chunker import semantic_chunk


DATASET_DIR = "dataset"
PDF_DIR = "data"
VECTOR_DB_DIR = "vector_db"

documents = []


# ------------------------
# Load Website Dataset
# ------------------------

print("Loading website dataset...")

for file in os.listdir(DATASET_DIR):

    if file.endswith(".md"):

        filepath = os.path.join(DATASET_DIR, file)

        loader = TextLoader(filepath, encoding="utf-8")

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file

        documents.extend(docs)


# ------------------------
# Load PDF Files
# ------------------------

print("Loading PDF documents...")

for file in os.listdir(PDF_DIR):

    if file.endswith(".pdf"):

        filepath = os.path.join(PDF_DIR, file)

        loader = PyPDFLoader(filepath)

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file

        documents.extend(docs)


print("Total documents loaded:", len(documents))


# ------------------------
# Semantic Chunking
# ------------------------

print("Creating semantic chunks...")

chunked_documents = []

for doc in documents:

    text = doc.page_content
    source = doc.metadata.get("source", "unknown")

    chunks = semantic_chunk(text)

    for chunk in chunks:

        chunked_documents.append(
            Document(
                page_content=chunk,
                metadata={"source": source}
            )
        )

print("Total chunks created:", len(chunked_documents))


# ------------------------
# Load Embedding Model
# ------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# ------------------------
# Create Vector Database
# ------------------------

print("Creating vector database...")

db = Chroma.from_documents(
    chunked_documents,
    embeddings,
    persist_directory=VECTOR_DB_DIR
)

db.persist()

print("Vector database created successfully!")
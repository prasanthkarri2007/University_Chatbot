from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

def search_documents(query):
    results = db.similarity_search(query, k=2)
    return results
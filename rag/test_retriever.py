from retriever import search_documents

query = input("Ask something: ")

docs = search_documents(query)

for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print(doc.page_content)
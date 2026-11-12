from tools import search_knowledge, list_documents, get_document

print("=== list_documents ===")
print(list_documents.invoke({}))

print("\n=== search_knowledge ===")
print(search_knowledge.invoke({"query": "my bench press stopped improving"}))

print("\n=== get_document (first 300 chars) ===")
print(get_document.invoke({"title": "ai_agents_explained"})[:300])
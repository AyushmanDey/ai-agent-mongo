import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.tools import tool
from embeddings import embed

load_dotenv()

_client = MongoClient(os.getenv("MONGODB_URI"))
_db = _client["ai_agents"]
chunked_docs = _db["chunked_docs"]
full_docs = _db["full_docs"]


@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base for passages relevant to a question.
    Use this to answer specific factual questions about any topic
    in the knowledge base.

    Args:
        query: The user's question or a search phrase.
    """
    query_vector = embed([query])[0]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,   # how many to consider
                "limit": 4,             # how many to return
            }
        },
        {
            "$project": {
                "_id": 0,
                "title": 1,
                "body": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = list(chunked_docs.aggregate(pipeline))
    if not results:
        return "No relevant information found."

    return "\n\n---\n\n".join(
        f"[Source: {r['title']} | score {r['score']:.2f}]\n{r['body']}" for r in results
    )


@tool
def list_documents() -> str:
    """List the titles of all documents in the knowledge base.
    Use this when the user asks what documents are available, or before
    calling get_document if you are not sure of the exact title."""
    titles = [d["title"] for d in full_docs.find({}, {"_id": 0, "title": 1})]
    return "\n".join(titles) if titles else "The knowledge base is empty."


@tool
def get_document(title: str) -> str:
    """Get the full text of one document by its exact title.
    Use this when the user wants a summary or overview of a whole document.

    Args:
        title: The exact document title (use list_documents to find it).
    """
    doc = full_docs.find_one({"title": title}, {"_id": 0, "body": 1})
    return doc["body"] if doc else f"No document titled '{title}'. Try list_documents."


ALL_TOOLS = [search_knowledge, list_documents, get_document]
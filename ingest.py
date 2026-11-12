import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from pypdf import PdfReader
from embeddings import embed, EMBEDDING_DIM

load_dotenv()
if not os.getenv("MONGODB_URI"):
    raise SystemExit("MONGODB_URI not found. Is your .env file in the project folder?")

KNOWLEDGE_DIR = Path("knowledge")
DB_NAME = "ai_agents"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks


def read_file(path: Path) -> str:
    """Read text from a PDF, TXT or MD file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks so no idea gets cut in half."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def create_vector_index(collection):
    """Create the Atlas Vector Search index (only if it doesn't exist yet)."""
    existing = [idx["name"] for idx in collection.list_search_indexes()]
    if "vector_index" in existing:
        print("Vector index already exists")
        return

    index = SearchIndexModel(
        name="vector_index",
        type="vectorSearch",
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": EMBEDDING_DIM,
                    "similarity": "cosine",
                }
            ]
        },
    )
    collection.create_search_index(model=index)
    print("Creating vector index (can take ~1 minute)...")

    while True:
        status = [i for i in collection.list_search_indexes() if i["name"] == "vector_index"]
        if status and status[0].get("queryable"):
            print("Vector index is ready")
            return
        time.sleep(5)


def main():
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[DB_NAME]
    full_docs = db["full_docs"]
    chunked_docs = db["chunked_docs"]

    # Start fresh each run so re-running doesn't create duplicates
    full_docs.delete_many({})
    chunked_docs.delete_many({})

    files = [p for p in KNOWLEDGE_DIR.iterdir() if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    if not files:
        print("No files found in knowledge/ — add some PDFs, TXT or MD files first.")
        return

    for path in files:
        text = read_file(path)
        title = path.stem   # file name without extension

        full_docs.insert_one({"title": title, "body": text})

        chunks = chunk_text(text)
        vectors = embed(chunks)
        chunked_docs.insert_many(
            [{"title": title, "body": c, "embedding": v} for c, v in zip(chunks, vectors)]
        )
        print(f"{title}: {len(chunks)} chunks")

    create_vector_index(chunked_docs)
    print("Done! Knowledge loaded into MongoDB.")


if __name__ == "__main__":
    main()
from sentence_transformers import SentenceTransformer

# Small, free model that runs on your PC. Outputs 384 numbers per text.
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed(texts):
    """Turn a list of strings into a list of vectors."""
    return get_model().encode(texts, normalize_embeddings=True).tolist()

EMBEDDING_DIM = 384
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embedding(text: str):
    """
    Convert a text string into a numerical vector.
    """
    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding


def generate_embeddings(texts: list[str]):
    """
    Convert multiple text chunks into vectors.
    """
    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings
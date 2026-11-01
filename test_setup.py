
from backend.services.embedding_service import generate_embedding
from backend.database.vector_store import get_collection_count

text = "How can I get my refund?"

embedding = generate_embedding(text)

print("Embedding generated successfully!")
print("Vector dimensions:", len(embedding))
print("First five values:", embedding[:5])

print("ChromaDB connected!")
print("Stored document count:", get_collection_count())

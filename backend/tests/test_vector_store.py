from backend.services.embedding_service import generate_embedding
from backend.services.vector_store import VectorStore

documents = [
    {"text": "Python is a programming language."},
    {"text": "Machine learning enables computers to learn from data."},
    {"text": "You can reset your password from account settings."},
    {"text": "Deep learning uses neural networks."}
]

store = VectorStore()

# Generate embeddings for documents
texts = [doc["text"] for doc in documents]
embeddings = [
    generate_embedding(text)
    for text in texts
]

store.add_documents(documents, embeddings)

# Search
query = "How do I change my account password?"
query_embedding = generate_embedding(query)

results = store.search(query_embedding, top_k=2)

print("\nSearch Results:")

for result in results:
    print("\nDocument:", result["document"]["text"])
    print("Similarity:", round(result["similarity_score"], 4))
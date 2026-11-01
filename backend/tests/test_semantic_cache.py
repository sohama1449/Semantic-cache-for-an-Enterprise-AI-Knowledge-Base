
from backend.services.semantic_cache import SemanticCache
from backend.services.embedding_service import generate_embedding

original_question = "How many sick leaves do employees receive?"

test_questions = [
    "How many sick leaves do employees receive?",
    "What is the annual sick leave allowance?",
    "How many casual leaves do employees receive?",
    "What are the company working hours?"
]

original_embedding = generate_embedding(original_question)

cache = SemanticCache(similarity_threshold=0.85)

cache.add(
    question=original_question,
    embedding=original_embedding,
    answer="Employees receive 12 sick leaves per year.",
    sources=[{"page": 1}]
)

print("\nSEMANTIC CACHE EVALUATION")
print("-" * 65)

for question in test_questions:

    query_embedding = generate_embedding(question)

    similarity = float(original_embedding @ query_embedding)

    result = cache.search(query_embedding)

    status = "HIT" if result else "MISS"

    print("\nQuestion:", question)
    print("Similarity:", round(similarity, 4))
    print("Cache Status:", status)

print("\nEvaluation completed.")


from backend.services.knowledge_base import KnowledgeBase

kb = KnowledgeBase()

pdf_path = "sample_documents/company_policy.pdf"

count = kb.add_pdf(pdf_path)

print("\nDocuments indexed:", count)

question = "How many sick leaves do employees receive?"

print("\n========== FIRST REQUEST ==========")

result1 = kb.ask_question(question)

print("Answer:", result1["answer"])
print("Cache Hit:", result1["cache_hit"])
print("Latency:", result1["latency_ms"], "ms")


print("\n========== SECOND REQUEST ==========")

result2 = kb.ask_question(question)

print("Answer:", result2["answer"])
print("Cache Hit:", result2["cache_hit"])
print("Latency:", result2["latency_ms"], "ms")


print("\n========== CACHE STATISTICS ==========")

print(kb.semantic_cache.get_stats())

print("\n========== SEMANTIC CACHE TESTS ==========")

test_questions = [
    "What is the annual sick leave allowance?",
    "How many casual leaves do employees receive?",
    "What are the company's working hours?"
]

for question in test_questions:
    result = kb.ask_question(question)

    print("\nQuestion:", question)
    print("Answer:", result["answer"])
    print("Cache Hit:", result["cache_hit"])
    print("Latency:", result["latency_ms"], "ms")

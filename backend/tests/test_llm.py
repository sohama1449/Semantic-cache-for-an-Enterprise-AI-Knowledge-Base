
from backend.services.llm_service import generate_answer

question = "How many sick leaves do employees receive?"

retrieved_results = [
    {
        "document": {
            "text": "Every employee receives 12 casual leaves and 12 sick leaves per calendar year.",
            "page_number": 1
        },
        "similarity_score": 0.85
    }
]

answer = generate_answer(question, retrieved_results)

print("\nQuestion:", question)
print("\nGenerated Answer:")
print(answer)

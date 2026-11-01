
from backend.services.knowledge_base import KnowledgeBase

kb = KnowledgeBase()

pdf_path = "sample_documents/company_policy.pdf"

# Step 1: Add PDF to knowledge base
count = kb.add_pdf(pdf_path)

print("\nTotal chunks indexed:", count)

# Step 2: Ask a question
question = "How many sick leaves do employees receive?"

print("\nQuestion:", question)

# Step 3: Search relevant information
results = kb.search_documents(question, top_k=3)

print("\nRetrieved Results:")

for result in results:
    print("\nRelevant Text:")
    print(result["document"]["text"])

    print("Page:", result["document"]["page_number"])
    print("Similarity:", round(result["similarity_score"], 4))

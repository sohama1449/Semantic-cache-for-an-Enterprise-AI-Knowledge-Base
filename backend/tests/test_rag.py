
from backend.services.knowledge_base import KnowledgeBase

# Initialize knowledge base
kb = KnowledgeBase()

# Upload and index PDF
pdf_path = "sample_documents/company_policy.pdf"

count = kb.add_pdf(pdf_path)

print("Documents indexed:", count)

# Ask a question
question = "How many sick leaves do employees receive?"

result = kb.ask_question(question)

print("\nQuestion:")
print(result["question"])

print("\nAI Generated Answer:")
print(result["answer"])

print("\nSources:")
for source in result["sources"]:
    print(
        f"Page: {source['page']}, "
        f"Similarity: {source['similarity']}"
    )

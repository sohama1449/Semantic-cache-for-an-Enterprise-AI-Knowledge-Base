from backend.services.embedding_service import generate_embedding

text = "Artificial Intelligence helps machines learn."

vector = generate_embedding(text)

print("Embedding generated successfully!")
print("Vector dimensions:", len(vector))
print("First 5 values:", vector[:5])
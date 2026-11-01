
import numpy as np


class VectorStore:

    def __init__(self):
        self.embeddings = []
        self.documents = []

    def add_documents(self, documents, embeddings):

        for document, embedding in zip(documents, embeddings):
            self.documents.append(document)
            self.embeddings.append(embedding)

    def search(self, query_embedding, top_k=3, min_similarity=0.30):

        if not self.embeddings:
            return []

        # Convert stored embeddings into a NumPy array
        vectors = np.array(self.embeddings)

        # Calculate cosine similarity
        scores = vectors @ query_embedding

        # Get indices of highest similarity scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:

            score = float(scores[index])

            if score >= min_similarity:

                results.append({
                    "document": self.documents[index],
                    "similarity_score": score
                })

        return results

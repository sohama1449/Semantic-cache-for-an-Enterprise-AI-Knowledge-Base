
import time

from backend.services.document_processor import process_pdf
from backend.services.embedding_service import generate_embeddings
from backend.services.vector_store import VectorStore
from backend.services.llm_service import generate_answer
from backend.services.semantic_cache import SemanticCache


class KnowledgeBase:

    def __init__(self):
        self.vector_store = VectorStore()

        self.semantic_cache = SemanticCache(
            similarity_threshold=0.85,
            ttl=3600
        )

    def add_pdf(self, file_path):
        chunks = process_pdf(file_path)

        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        embeddings = generate_embeddings(texts)

        self.vector_store.add_documents(chunks, embeddings)

        return len(chunks)

    def search_documents(self, question, top_k=3):
        query_embedding = generate_embeddings([question])[0]

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        return results

    def ask_question(self, question, top_k=3):

        start_time = time.time()

        # 1. Convert question into an embedding
        query_embedding = generate_embeddings([question])[0]

        # 2. Check semantic cache first
        cached_result = self.semantic_cache.search(
            query_embedding,
            question
        )

        # 3. If cache hit, return stored answer
        if cached_result:

            return {
                "question": question,
                "answer": cached_result["answer"],
                "sources": cached_result["sources"],
                "cache_hit": True,
                "similarity": round(
                    cached_result["similarity_score"], 4
                ),
                "latency_ms": round(
                    (time.time() - start_time) * 1000, 2
                )
            }

        # 4. Cache miss: perform document retrieval
        retrieved_results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        # 5. Generate grounded answer using Groq
        answer = generate_answer(
            question,
            retrieved_results
        )

        # 6. Prepare source references
        sources = [
            {
                "page": result["document"]["page_number"],
                "similarity": round(
                    result["similarity_score"], 4
                )
            }
            for result in retrieved_results
        ]

        # 7. Store answer in semantic cache
        self.semantic_cache.add(
            question=question,
            embedding=query_embedding,
            answer=answer,
            sources=sources
        )

        # 8. Return response
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "cache_hit": False,
            "similarity": None,
            "latency_ms": round(
                (time.time() - start_time) * 1000, 2
            )
        }

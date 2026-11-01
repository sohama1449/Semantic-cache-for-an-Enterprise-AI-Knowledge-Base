
import time
import re


class SemanticCache:

    def __init__(self, similarity_threshold=0.85, ttl=3600):

        self.cache = []
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl

        self.hits = 0
        self.misses = 0

        # Common words that do not add much meaning
        self.stopwords = {
            "how", "what", "when", "where", "who",
            "is", "are", "the", "a", "an", "do",
            "does", "did", "to", "of", "in", "on",
            "for", "per", "and", "from", "employees"
        }

        # Concepts that should not be confused
        self.conflicting_terms = [
            {"sick", "casual"},
            {"salary", "leave"},
            {"salary", "working"},
        ]

    # 1. Normalize question keywords
    def normalize_keywords(self, question):

        words = re.findall(r"[a-z]+", question.lower())
        normalized = set()

        for word in words:
            if word.endswith("ies"):
                word = word[:-3] + "y"
            elif word.endswith("s") and len(word) > 3:
                word = word[:-1]

            if word not in self.stopwords:
                normalized.add(word)

        return normalized

    # 2. Extract numerical values
    def extract_numbers(self, question):

        return set(
            re.findall(r"\b\d+(?:\.\d+)?\b", question.lower())
        )

    # 3. Detect negation
    def has_negation(self, question):

        negation_terms = {
            "no", "not", "never", "cannot", "can't",
            "don't", "doesn't", "isn't", "aren't",
            "won't", "prohibited", "forbidden",
            "without", "disallowed"
        }

        words = set(re.findall(r"[a-z']+", question.lower()))

        return bool(words.intersection(negation_terms))

    # 4. Check whether two questions are safe to match
    def is_safe_match(self, old_question, new_question):

        old_words = self.normalize_keywords(old_question)
        new_words = self.normalize_keywords(new_question)

        # Reject questions with no meaningful keyword overlap
        if not old_words.intersection(new_words):
            return False

        # Reject different or missing numerical values
        old_numbers = self.extract_numbers(old_question)
        new_numbers = self.extract_numbers(new_question)

        if old_numbers != new_numbers:
            return False

        # Reject opposite negation/polarity
        old_negation = self.has_negation(old_question)
        new_negation = self.has_negation(new_question)

        if old_negation != new_negation:
            return False

        # Reject conflicting concepts
        for pair in self.conflicting_terms:

            if pair.issubset(old_words.union(new_words)):

                if (
                    pair.intersection(old_words)
                    and pair.intersection(new_words)
                    and old_words != new_words
                ):
                    return False

        return True

    # 5. Add a question-answer pair to cache
    def add(self, question, embedding, answer, sources=None):

        self.cache.append({
            "question": question,
            "embedding": embedding,
            "answer": answer,
            "sources": sources or [],
            "timestamp": time.time()
        })

    # 6. Search semantic cache
    def search(self, query_embedding, query_text):

        current_time = time.time()

        # Remove expired cache entries
        self.cache = [
            item for item in self.cache
            if current_time - item["timestamp"] < self.ttl
        ]

        best_match = None
        best_score = -1

        # Compare query embedding with cached embeddings
        for item in self.cache:

            score = float(item["embedding"] @ query_embedding)

            print(
                f"Cache comparison: '{item['question']}' "
                f"| Similarity: {score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_match = item

        # Perform safety check
        safety_check = (
            self.is_safe_match(
                best_match["question"],
                query_text
            )
            if best_match else False
        )

        # Debugging information
        print("\n----- CACHE DEBUG INFORMATION -----")
        print("Best similarity:", round(best_score, 4))
        print("Threshold:", self.similarity_threshold)
        print("Safety check:", safety_check)

        # Cache hit condition
        if (
            best_match
            and best_score >= self.similarity_threshold
            and safety_check
        ):

            self.hits += 1

            print("Cache Result: HIT")

            return {
                "question": best_match["question"],
                "answer": best_match["answer"],
                "sources": best_match["sources"],
                "similarity_score": best_score,
                "cache_hit": True
            }

        # Cache miss
        self.misses += 1

        print("Cache Result: MISS")

        return None

    # 7. Cache statistics
    def get_stats(self):

        total = self.hits + self.misses

        hit_rate = (
            self.hits / total * 100
            if total > 0 else 0
        )

        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2)
        }

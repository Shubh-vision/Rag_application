from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from chunk_retriever import embedding, index

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

SIMILARITY_THRESHOLD = 0.70

semantic_cache = []

def get_cached_answer(query):

    print("Cache Size:", len(semantic_cache))

    if len(semantic_cache) == 0:
        return None
    
    query_embedding = embedding_model.encode(query)

    best_score = 0
    best_answer = None

    for item in semantic_cache:

        score = cosine_similarity(
            [query_embedding],
            [item['embedding']]
        )[0][0]

        if score > best_score:
            best_score = score
            best_answer = item["answer"]

        
        print(f"Best Cache Similarity: {best_score:.3f}")

        if best_score >= SIMILARITY_THRESHOLD:
            print("Semantic Cache HIT")
            return best_answer
        
        return None
    


def save_to_cache(query, answer):

    embedding = embedding_model.encode(query)

    semantic_cache.append(
        {
            "query": query,
            "embedding": embedding,
            "answer": answer
        }
    )

    print("Saved To Semantic Cache")


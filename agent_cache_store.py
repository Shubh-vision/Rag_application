from semantic_cache import save_to_cache
from chunk_retriever import embedding, index
import uuid



def cache_store_agent(state):

    query = state["query"]
    answer = state["answer"]['answer']
    vector = embedding.embed_query(query)

    index.upsert(
        vectors=[
            {
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": {
                    "query": query,
                    "answer": answer
                }
            }
        ],
        namespace="cache"
    )


    print("Saved To Semantic Cache")

    return state
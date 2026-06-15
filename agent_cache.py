from state import AgentState
from chunk_retriever import embedding, index


CACHE_THRESHOLD = 0.85


def cache_agent(state : AgentState):
    print("Entered Cache")

    query = state["query"]

    vector = embedding.embed_query(query)

    result = index.query(
        vector=vector,
        top_k=1,
        namespace="cache",
        include_metadata=True
    )

    print(result)

    if result.matches:

        best_match = result.matches[0]

        score = best_match.score

        print(f"Similarity = {score}")

        if score >= CACHE_THRESHOLD:

            print("CACHE HIT")

            return {
                **state,
                "cache_hit": True,
                "answer": {
        "answer": best_match.metadata["answer"]},
        
                "source" : "cache"   
                        }

    print("CACHE MISS")

    return {
        **state,
        "cache_hit": False,
        "source" : "rag" 
    }



def route_cache(state : AgentState):

    if state["cache_hit"]:
        return "store"

    return "retriever"


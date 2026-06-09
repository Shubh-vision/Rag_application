import supabase


def store_to_supabase(state):

    supabase.table("rag_answers").insert({
        "query": state["query"],
        "answer": state["answer"],
        "model_used" : state['model_used'],
        "token_usage" : state['token_usage'],
        "source": state.get("source", "unknown")
    }).execute()

    return state
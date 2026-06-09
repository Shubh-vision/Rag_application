from db_config import get_connection



def store_postgres(state):

    conn = get_connection()

    if conn is None:
        print("❌ DB connection failed, skipping insert")
        return state
    
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO rag_answers
        (query, answer, summary, model_used, token_usage)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        state["query"],
        state["answer"]["answer"],
        state.get("summary", ""),               # if you don't have summary yet → empty string
        state.get("model_used", ""),
        state.get("token_usage", 0)
    ))

    conn.commit()
    cur.close()
    conn.close()

    return state
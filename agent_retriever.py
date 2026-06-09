from state import AgentState
from chunk_retriever import retriever

from langsmith import traceable
import os

print(os.getenv("LANGSMITH_TRACING"))
print(os.getenv("LANGSMITH_PROJECT"))


@traceable(name = "Retriever")
def retriever_agent(state : AgentState):
    print("Entered Retriever")
    query = state['query']

    document = retriever.invoke(query)

    return {
        **state, "document": document
    }



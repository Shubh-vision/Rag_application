from langgraph.graph import StateGraph, START, END
from state import AgentState
from agent_evaluator import evaluator_agent, route_after_evaluation
from agent_generate import generate_agent
from agent_retriever import retriever_agent
from langgraph.checkpoint.memory import InMemorySaver
from agent_cache import cache_agent, route_cache
from agent_cache_store import cache_store_agent
from store_postgres import store_postgres 





builder = StateGraph(AgentState)

#Define Node

builder.add_node("retriever", retriever_agent)
builder.add_node("generate", generate_agent)
builder.add_node("evaluator", evaluator_agent)
builder.add_node("cache", cache_agent)
builder.add_node("cache_store",cache_store_agent)
builder.add_node("postgres_store", store_postgres)

# Define edges
builder.add_edge(START,"cache")

builder.add_conditional_edges(
    "cache",
    route_cache,
    {
        # "approved": END,
        "store": "cache_store",
        "retriever": "retriever"
    }
)

builder.add_edge("retriever", "generate")
builder.add_edge("generate", "evaluator")


#Conditional node

builder.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {
        "retriever": "retriever",
        "generate": "generate",
        "approved": "cache_store"
    }
)



builder.add_edge("cache_store", "postgres_store")
builder.add_edge("postgres_store", END)



graph = builder.compile()



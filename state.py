from pydantic import BaseModel
from typing import TypedDict, Dict, Any

# State for Evaluator

class Evaluator(BaseModel):
    relevant : bool
    grounded : bool
    answer_question : bool

# Define State
class AgentState(TypedDict):
    query: str
    document: list
    answer: str
    context: str
    result: Evaluator
    model_used: str
    token_usage: int
    cache_hit: bool
    retriever: Any
    





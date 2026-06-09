from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
from state import Evaluator, AgentState
from langsmith import traceable
import os
 
print(os.getenv("LANGSMITH_TRACING"))
print(os.getenv("LANGSMITH_PROJECT"))


# LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

structured_llm = llm.with_structured_output(Evaluator, method="json_mode")

System = """
You are an expert evaluator.

Your task is to determine whether the answer is fully supported
by the provided context.

Return ONLY valid JSON.

{{
   "relevant": boolean,
  "grounded": boolean,
  "answer_question": boolean
}}

context:
{context}

query:
{query}

answer:
{answer}
"""

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', System),
        ('human', "context:\n{context}\nquery:\n{query}\n\nanswer:\n{answer}")
    ]
)

evaluator = grader_prompt | structured_llm


@traceable(name="Evaluator", tags=["rag", "evaluation"])
def evaluator_agent(state : AgentState):
    print("Entered Evaluator")
    answer_text = state["answer"]["answer"]
    query = state['query']
    document = state['document']

    context = "\n\n".join([d.page_content for d in document])

    result = evaluator.invoke({
        "query" : query, 
        'answer' : answer_text,
        "context": context
    })

    return {**state, "result": result, "answer": state["answer"]}





@traceable(name="Router")
def route_after_evaluation(state: AgentState):
    print("Entered router")

    result = state['result']

    if not result.relevant:
        return "retriever"
    
    if not result.grounded:
        return "retriever"
    
    if not result.answer_question:
        return "generate"

    return "approved"

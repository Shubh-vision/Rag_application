from state import AgentState
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from llmlite import llm
from langsmith import traceable
load_dotenv()
import os

#the Generate Agent should:

#Receive the user question.
#Receive retrieved documents from the Retriever Agent.
#Build the context.
#Send the prompt to the LLM.
#Return the final answer.

#llm
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

print(os.getenv("LANGSMITH_TRACING"))
print(os.getenv("LANGSMITH_PROJECT"))

System = """You are a helpful AI assistant.

Provide a complete and detailed answer.
Include all relevant information from the context.
Do not summarize unless the user asks for a summary.

If the answer is not found in the context,
say:
"I could not find the answer in the provided documents."

Context:
{context}

query:
{query}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", System),
        ("human", "Context:\n{context}\nQuery:\n{query}")
    ]
)

generate_model = prompt | llm 


@traceable(name="Generator", tags=["rag", "generation"])
def generate_agent(state : AgentState):
    print("Entered generate")

    query = state['query']
    document = state['document']

    if not document:
        return {**state, "answer": "I don't know based on the available information."}
    
    context = "\n\n".join([d.page_content for d in document])

    try:
        # print(context)
        answer = generate_model.invoke({"context": context, "query" : query})
        print(f"Retrieved docs: {len(document)}")

        return {**state, "answer": answer, 
                "model_used": answer['model'],
                "token_usage": answer['tokens']
                }
       

    except Exception as e:
        print("\n========== GENERATOR ERROR ==========")
        print(type(e))
        print(e)
        raise e



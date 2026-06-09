from litellm import completion
from litellm import completion_cost
from langchain_core.runnables import RunnableLambda
import os

from dotenv import load_dotenv
load_dotenv()


def call_llm(prompt):

    prompt_text = prompt.to_string()

    response = completion(
        model = "mistral/mistral-medium-latest",


        #Automatic fallback chain
        fallbacks = [
            "gemini/gemini-2.5-flash",
            "mistral/mistral-small-latest",
            "mistral/mistral-medium-latest",
            "groq/llama-3.3-70b-versatile",
            "mistral/open-mistral-7b",
            "mistral/open-mixtral-8x22b"
        ],

        messages=[
            {
                "role" : "user",
                "content" : prompt_text
            }
        ],

        temperature=0,
        max_tokens=1000
    )
    print("MODEL =", response.model)
    print(f"\nModel Used: {response.model}")
    print(f"Tokens: {response.usage.total_tokens}")
    

    return {
    "answer": response.choices[0].message.content,
    "model": response.model,
    "tokens": response.usage.total_tokens,
    }


llm = RunnableLambda(call_llm)

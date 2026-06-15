from graph import graph
from ingestion import get_docs

def main():

    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        result = graph.invoke(
        {
            "query" : query,
            "document" : [],
            "answer" : "",
            "result" : None,
            "context": "",
            "model_used": "",
            "token_usage": 0
        })

    

        evaluation = result["result"]
        cache_hit = result.get("cache_hit", False)

        print("\n========== ANSWER ==========\n")
        print(result["answer"]["answer"])

        print("\n========== INFO ==========\n")

        if cache_hit:
            print("Source           : Semantic Cache")
            print("Evaluator        : Skipped")
        else:
            print("Source           : RAG Pipeline")
            print(f"Relevant         : {evaluation.relevant}")
            print(f"Grounded         : {evaluation.grounded}")
            print(f"Answer Question  : {evaluation.answer_question}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting via Ctrl+C...")
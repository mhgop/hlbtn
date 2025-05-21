# ------------------------------------------------- Provided Code -------------------------------------------------
import os
import logging
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)

def llm_call(prompt):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=16384,
        temperature=0
    )
    return response.choices[0].message.content

def fallback_response() -> str:
    return "The query could not be classified. Please refine your query or contact support."

# ------------------------------------------------- Code to be completed -------------------------------------------------

def safe_llm_call(prompt):
    try:
        return llm_call(prompt)
    except Exception as e:
        logging.error(f"LLM Call Error: {e}")
        return "An error occurred while processing your request."

def classify_query(query:str) -> str:
    # Create a prompt for the classification
    classification_prompt = ""
    # call the llm and normalize the output
    response = ""
    # return the response if the the task is within the scope
    # return unknown otherwise

def handle_translation(query:str, target_language:str) -> str:
    # create a translation prompt and call the llm
    return 

def handle_question(query:str) -> str:
    # create a question answering prompt and call the llm
    return 

TASK_HANDLERS = {
    "translation": handle_translation,
    "question answering": handle_question,
}

def dispatch_task(task_type:str, query:str, **kwargs) -> str:
    handler = TASK_HANDLERS.get(task_type)
    if handler:
        return handler(query, **kwargs)
    else:
        return "Unknown task type."


def main():
    print("\nWelcome to the Multi-Step Dispatcher!")
    print("You can ask a question or request a translation.\n")

    query = input("Enter your query: ")

    logging.info("Classifying query...")
    query_type = ""

    if "translation" in query_type:
        # retrieve the target language
        target_language = ""
        # use the dispatch task function
        result = ""
    elif query_type == "question answering":
        # use the dispatch task function
        result = ""
    else:
        # use the fallback
        result = ""

    print("\nResult:")
    print(result)

if __name__ == "__main__":
    main()
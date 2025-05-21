# ------------------------------------------------- Provided Code -------------------------------------------------
import os
from openai import OpenAI

# Initialize the OpenAI client
def llm_call(prompt:str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0
    )
    return response.choices[0].message.content.strip()

def human_hand_over(query: str) -> str:
    """Handles cases where human intervention is required."""
    return f"Unable to process the query: {query}. Please consult a human for further assistance."

# ------------------------------------------------- Code to be completed -------------------------------------------------
def ai_dispatcher(query:str) -> str:
    # Create a prompt for the classification
    # call llm with this prompt
    return 

def translation_agent(query:str) -> str:
    # Create a prompt for the classification
    # call llm with this prompt
    return

def qa_agent(query:str) -> str:
    # Create a prompt for the classification
    # call llm with this prompt
    return

def summary_agent(query:str) -> str:
    # Create a prompt for the classification
    # call llm with this prompt
    return

def process_query(query: str) -> str:
    """
    Processes the query based on its intent and returns the result.
    """
    try:
        # Determine intent
        intent = ""

        # Mapping of intents to agent functions
        intent_to_agent = {}

        # Handle intent if within the scope, else hand over to human

    except Exception as e:
        return f"An error occurred while processing the query: {str(e)}"

def main():
    print("Welcome to the Hand-Over Example!")
    while True:
        # Get user query
        query = input("\nEnter your query, type 'exit' to quit the app: ").strip()
        if query.lower() == "exit":
            print("Thank you for using the AI Hand-Over System. Goodbye!")
            break

        # Process query
        print("\nProcessing your query with AI...")
        response = ""
        
        # Print the response
        print("\nAI Response:")
        print(response)

if __name__ == "__main__":
    main()

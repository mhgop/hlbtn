# ------------------------------------------------- Provided Code -------------------------------------------------
import os
from openai import OpenAI

# Initialize the OpenAI client
def llm_call(prompt):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ------------------------------------------------- Code to be completed -------------------------------------------------

def main():
    print("Welcome to the Step Dispatcher!")
    print("You can either request a translation or a math calculation.")
    
    # User Input
    task_type = input("Enter the task type (translation/math): ").strip().lower()
    query = input("Enter your input: ").strip()
    
    # Dispatcher Logic
    if task_type == "translation":
        # Input the target language
        target_language = ""
        # create a prompt and call the llm
        result = ""
    elif task_type == "math":
        # create a prompt and call the llm
        result = ""
    else:
        result = ""
    
    # Output the result
    print("\nResult:")
    print(result)

if __name__ == "__main__":
    main()

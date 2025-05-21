# --- --- --- PROVIDED CODE --- --- ---
import os
import requests
import giskard
import pandas as pd
from openai import OpenAI
from giskard import scan, Dataset, Model
from giskard.llm.client.openai import OpenAIClient

openai_client = OpenAIClient(model="gpt-4o-mini")
giskard.llm.set_llm_api("openai")
giskard.llm.set_default_client(openai_client)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---
# TODO
# Pick an url of a Wikipedia page on the topic you like.
url = ""

def get_wikipedia_context(url: str) -> str:
    """Download and return the content of the Wikipedia page."""
    # Use the requests library to download the Wikipedia page content.
    # If needed slice the content to make sure the model's limitations are not exceeded.
    # Raise an exception if the request fails.
    return 

def make_prompt(question:str, context: str) -> str:
    """Combine a given question and context to form a prompt for an AI model."""
    return 

def ask(question: str, model: str) -> str:
    """Answers a query using an AI model."""
    # Choose any model and generate its response on a given question.
    # Use a system prompt and add the context either inside the system or inside the user prompt.
    return 

def prediction_function(df: pd.DataFrame) -> list[str]:
    """Generate answers to the questions using the provided DataFrame of relevant texts."""
    # return a list of answers to the questions from the DataFrame using the ask function.
    return 


def main():
    # TODO
    questions = [
    # Write questions on the topic you choose.
    ]
    raw_data = pd.DataFrame(data={"text": questions})
    dataset = Dataset(raw_data, target=None)
    # TODO
    model = Model(model=prediction_function, 
                    model_type='text_generation', 
                    feature_names=['text'],
                    name="",  # Provide a name for the model.
                    description="" # Provide a description for the model.
                )
    results = scan(model, dataset)
    # TODO
    results.save("") # Provide a path to save the results.

if __name__=="__main__":
    main()
# --- --- --- PROVIDED CODE --- --- ---
import os
import giskard
import requests
import pandas as pd
from openai import OpenAI
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

# LlamaIndex imports
from llama_index.core import VectorStoreIndex
from llama_index.core.data_structs import Node

# Giskard RAG evaluation imports
from giskard.rag import evaluate
from giskard.rag import AgentAnswer
from giskard.rag import KnowledgeBase, generate_testset
from giskard.rag.metrics.ragas_metrics import ragas_context_recall, ragas_context_precision

from giskard.llm.client.openai import OpenAIClient
openai_client = OpenAIClient(model="gpt-4o-mini")
giskard.llm.set_llm_api("openai")
giskard.llm.set_default_client(openai_client)

# Initialize the OpenAI client and make sure to set your OPENAI_API_KEY.
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

def split_text(text:str, 
               max_tokens:int=500) -> list[str]:
    """Splits text into chunks of a maximum number of tokens."""
    # Tokenize the text into sentences using nltk's sent_tokenize function.
    # Split each sentence into words using split() function.
    # Combine the words to form chunks of maximum length max_tokens.
    # Append each chunk to the chunks list.
    # If there are leftover words, append them as a single chunk.
    # Return the chunks list.
    return

def create_nodes(chunks:list[str]) -> list[Node]:
    """Creates Nodes from text chunks."""
    # Create a Node for each chunk of text and return the list of Nodes.
    return


def build_vector_index(nodes: list[Node]) -> VectorStoreIndex:
    """Builds a VectorStoreIndex from Nodes."""
    # Create a VectorStoreIndex from the list of Nodes. (Simple initialization)
    return

def retrieve_relevant_nodes(query:str, 
                            index:VectorStoreIndex, 
                            k:int=5) -> list[Node]:
    """Retrieves the most relevant Nodes for a given query."""
    # Use the index's as_query_engine method.
    # Query the index using the given query and retrieve the top-k relevant Nodes.
    return

def create_knowledge_base(nodes: list[Node]) -> KnowledgeBase:
    """Creates a KnowledgeBase object from Nodes."""
    # Create a DataFrame with the text content of the Nodes. Specify the column name as "text".
    # Return the KnowledgeBase object.
    return 
    
def make_prompt(question:str, 
                context: str) -> str:
    """Combine a given question and context to form a prompt for an AI model."""
    return 

def generate_answer(query:str, 
                    relevant_nodes: list[Node]) -> str:
    """Generates an answer to the query using the relevant Nodes as context."""
    # Choose any model and generate its response on a given question.
    # Use a system prompt and add the context either inside the system or inside the user prompt.
    return

def answer_fn(question:str, 
              index:VectorStoreIndex) -> AgentAnswer:
    """Function to answer questions using the provided index."""
    # Use the provided index to retrieve relevant Nodes for the given question.
    # Generate an answer using the relevant Nodes as context.
    # Return an AgentAnswer object with the generated answer and retrieved documents.
    # Return the AgentAnswer object.
    return


def main():
    context = get_wikipedia_context(url)
    if context:
        print("Splitting text into chunks...")
        chunks = split_text(context)
        print(f"Total chunks created: {len(chunks)}")

        print("Creating Nodes...")
        nodes = create_nodes(chunks)

        print("Building VectorStoreIndex...")
        index = build_vector_index(nodes)

        print("Creating Knowledge base...")
        knowledge_base = create_knowledge_base(nodes)

        print("Generating test set...")
        # TODO
        testset = generate_testset(knowledge_base,
                                   num_questions=None, # Provide the number of questions to evaluate. (Type: int)
                                   agent_description="" # Provide a description for the agent.
                                   )

        # Perform evaluation
        report = evaluate(
            lambda question: answer_fn(question, index),
            testset=testset,
            knowledge_base=knowledge_base,
            metrics=[ragas_context_recall, ragas_context_precision]
        )
        report.save("") # Provide a path to save the evaluation report.

if __name__ == "__main__":
    main()

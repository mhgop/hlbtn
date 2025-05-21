# ------------------------------------------------- Provided Code -------------------------------------------------
import os
import faiss
import cohere
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define paths for FAISS index and metadata
INDEX_FILE = "../data/vector_index.faiss"
METADATA_FILE = "../data/metadata.csv"


def llm_call(prompt:str, 
             model:str="gpt-4o-mini", 
             max_tokens:int=512, 
             temperature:float=0):
    """
    Makes a call to the LLM with the given prompt.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

# ------------------------------------------------- Code to be completed -------------------------------------------------
# Step 1: Load and Preprocess Dataset
def load_and_preprocess_dataset(dataset_path:str) -> dict:
    # Pick the columns to load after taking a look at the .csv file
    columns_to_load = []
    
    # Read the dataset, selecting only the specified columns
    df = ""
    
    # Convert the selected columns into a list of dictionaries (row-wise records)
    return df.to_dict(orient="records")


# Step 2: Embed Documents and Save to FAISS
def embed_and_store_in_faiss(documents:dict, 
                             model:str, 
                             index_file:str, 
                             metadata_file:str):
    # Use .encode method to embed
    embeddings = ""
    # define the second dimension using .shape
    dimension = ""

    # Create the index and add the embeddings to it
    # Look at FAISS docs to do so
    index = ""
    
    # Write the index on the index file using .write_index method

    # Create a metadata dataframe
    metadata_df = ""

# Step 3: Load FAISS and Metadata
def load_faiss_and_metadata(index_file:str, 
                            metadata_file:str) -> tuple:
    # Read index and csv from local files
    index = ""
    metadata_df = ""
    
    # Deserialize the `document` column (convert strings back to dictionaries) using apply eval
    metadata_df["document"] = ''
    return index, metadata_df

# Step 4: Retrieve Top-k Candidates
def retrieve_top_k(query:str, 
                   model:str, 
                   faiss_index, 
                   metadata_df:pd.DataFrame, 
                   top_k:int=10):
    # Embed the query, if necessary use .reshape method
    query_embedding = ""

    # Use .search method on the previously loaded index
    _, indices = ""
    
    # Retrieve the top documents and flatten them into strings
    # Format it as much as necessary
    top_documents = ""
    return 

# Step 5: Rerank Candidates using Cohere Rerank
def rerank_with_cohere(query:str, 
                       candidates:list, 
                       endpoint_url:str, 
                       api_key:str, 
                       top_n:int=5):
    # Initalize cohere client 
    co = cohere.Client(
                    base_url=endpoint_url, 
                    api_key=api_key
                    )
    # Use rerank method
    response = ""

    # Rerank based on the recovered indices
    sorted_indices = ""
    # Return the reranked candidates
    return 

    
# Step 6: Generate Clean Answer
def generate_answer(query:str, 
                    context:list):
    # Create a prompt and call the llm on it.
    prompt = ""
    return 

# Main Execution
if __name__ == "__main__":
    # Load SentenceTransformer model
    # Feel free to use any other embeddings model.
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Only create the embeddings if not already created.
    if not os.path.exists(INDEX_FILE):
        # Load dataset
        dataset_path = "../data/imdb_top_1000.csv"
        documents = load_and_preprocess_dataset(dataset_path)
        
        # Embed and store in FAISS
        embed_and_store_in_faiss(documents, 
                                 embedding_model, 
                                 INDEX_FILE, 
                                 METADATA_FILE)
    
    # Load FAISS index and metadata
    faiss_index, metadata_df = load_faiss_and_metadata(INDEX_FILE, 
                                                       METADATA_FILE)
    
    # User query
    user_query = input("Ask a question about a famous movie: \n")
    
    # Retrieve top candidates
    top_candidates = retrieve_top_k(user_query, 
                                    embedding_model, 
                                    faiss_index, 
                                    metadata_df, 
                                    top_k=10)
    
    # Rerank candidates with Cohere
    reranked_results = rerank_with_cohere(user_query, 
                                          top_candidates, 
                                          os.environ["AZURE_ENDPOINT_URL"], 
                                          os.environ["AZURE_API_KEY"], 
                                          top_n=5)

    # Generate clean answer
    answer = generate_answer(user_query, reranked_results)
    print("\nGenerated Answer:")
    print(answer)

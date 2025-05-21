# --- --- --- PROVIDED CODE --- --- ---
import os
import mlflow
import giskard
import tiktoken
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

mlflow.set_tracking_uri('http://localhost:5000')

model_name = "gpt-4o-mini"

# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---
# TODO
# Chose one of the pdf file from the list below. Feel free to add more if needed.
pdf_url = ""
# "https://aiindex.stanford.edu/wp-content/uploads/2024/04/HAI_2024_AI-Index-Report.pdf"
# "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf"
# "https://wellbeing.hmc.ox.ac.uk/wp-content/uploads/2023/04/WHR23.pdf"
loader = PyPDFLoader(
    pdf_url
)

# TODO
# Prompt Template 
PROMPT_TEMPLATE = """ 

Context:
{context}

Question:
{question}

Your answer:
"""

# TODO
# DataFrame and Evaluation
df_example = pd.DataFrame({
    "query": [
        # Write questions related to the chosen PDF file.
    ]
})

# TODO
evaluator_config = {
    "model_config": {
        "name": "", # Provide a name.
        "description": "", # Provide a description.
        "feature_names": ["query"],
    },
}


def count_nb_tokens(text:str)-> int:
    """Count the number of tokens in a given text using tiktoken."""
    # Tokenize the text into tokens using tiktoken.
    # Count the number of tokens by getting the length of the encoded list.
    # Return the number of tokens.
    return
def load_and_split_pdf(loader):
    """Load the PDF file, split it into fragments, and load the documents into a vector store."""
    # Load the PDF file, split it into fragments, and load the documents into a vector store.
    # Use RecursiveCharacterTextSplitter to split the document into chunks.
    # Use the count_nb_tokens function to determine the length of each chunk.
    # Return the splitted list of documents.
    return

def load_into_vector_db(docs):
    """Load the documents into a FAISS vector store."""
    # Load the documents into a FAISS vector store.
    # Use the OpenAIEmbeddings model to embed the documents into a vector space.
    # Create a FAISS vector store from the documents and the embedded vectors.
    return 

def evaluate_model(df, retrieval_chain):
    """Evaluate the model using the provided DataFrame and retrieval chain."""
    # Evaluate the model using the provided DataFrame and retrieval chain.
    # Use the retrieval chain to retrieve the answers for each question in the DataFrame.
    return 

def get_llm_and_retrieval_chain(model_name, prompt):
    """Initialize the language model and retrieval chain for a given model."""
    # Initialize the language model and retrieval chain for a given model using ChatOpenAI.
    # Use the provided prompt template to create the retrieval chain. RetrievalQA.from_llm
    return

def save_report(report_dir, model_name, evaluation_result):
    """Save the evaluation result and report as JSON files."""
    # Save the evaluation result and report as JSON files.
    # Use the provided report directory and model name to create the file paths.
    # Save the evaluation result as a JSON file.
    # Log the report file as an MLflow artifact.
    # This function should not return anything.
    return None

# --- --- --- PROVIDED CODE --- --- ---
if __name__=="__main__":
    # Document Splitting
    print("Splitting the document into fragments...")
    docs = load_and_split_pdf(loader)

    print("Loading documents into FAISS vector store...")
    db = load_into_vector_db(docs)

    print("Initializing prompt template...")
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, 
                            input_variables=["question", "context"])

    print("Initializing language models and retrieval chains...")
    llm, retrieval_chain = get_llm_and_retrieval_chain(model_name, prompt)

    report_dir = "mlflow_reports"
    os.makedirs(report_dir, exist_ok=True)

    print(f"Running evaluation for model: {model_name}")
    with mlflow.start_run(run_name=model_name):
        evaluation_result = mlflow.evaluate(
            model=lambda df: evaluate_model(df, retrieval_chain),
            model_type="question-answering",
            data=df_example,
            evaluators="giskard",
            evaluator_config=evaluator_config,
        )

        save_report(report_dir, model_name, evaluation_result)
        
# --- --- --- PROVIDED CODE --- --- ---
import openai
import mlflow
import pandas as pd
from datasets import load_dataset
from mlflow.metrics.genai import EvaluationExample, make_genai_metric

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://0.0.0.0:5000")

# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---
def load_dataset_from_name(name:str, 
                           sample_size: int = 10):
    """Load a dataset from a given name and sample a specified number of rows."""
    # Choose and load a QA dataset
    # Sample this dataset with the input sample size using the `select` function
    return 

def preprocess_dataset(sampled_dataset) -> pd.DataFrame:
    """Preprocess the dataset by selecting relevant columns."""
    # The newly created DataFrame should contain two columns: "inputs" and "ground_truth" each containing a list of strings.
    return 

def load_qa_model(model_name: str):
    """Load a pre-trained OpenAI GPT-4o-mini model."""
    # Set a system prompt to guide the GPT-4o-mini model during evaluation.
    # Return the logged model using the `mlflow.openai.log_model` function.
    system_prompt = ""
    return 

def new_metric() -> mlflow.entities.Metric:
    """Create a new metric for evaluating a new aspect of the generated text."""
    return make_genai_metric(
        name="", # Choose a name for your new metric
        definition=(
            "" # Provide an explaination for this metric
        ),
        grading_prompt=(
            ""
            "- Score 1: "
            "- Score 2: "
            "- Score 3: "
            "- Score 4: "
            "- Score 5: "
            # Describe the scoring method for your metric.
        ),
        examples=[
            # Provide as many examples as you wish for your metric. Each example should include an input, a ground truth 'output', and a score.
            EvaluationExample(
                input="",
                output=(
                    ""
                ),
                score=2,
                justification=(
                    ""
                ),
            )
        ],
        version="v1",
        model="openai:/gpt-4o-mini",
        parameters={"temperature": 0.0},
        grading_context_columns=[],
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True
    )


# --- --- --- PROVIDED CODE --- --- ---
if __name__ == "__main__":
    print("Starting MLflow run...")
    with mlflow.start_run() as run:
        eval_df = preprocess_dataset(load_dataset_from_name("squad", 
                                                            sample_size=100))
        basic_qa_model = load_qa_model("gpt-4o-mini")
        print("Logged MLflow model...")
        results = mlflow.evaluate(
            basic_qa_model.model_uri,
            eval_df,
            targets="ground_truth",
            model_type="question-answering",
            evaluators="default",
            extra_metrics=[new_metric()]
        )
    print(results.tables["eval_results_table"])
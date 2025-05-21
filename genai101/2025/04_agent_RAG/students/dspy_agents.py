# ------------------------------------------------- Provided Code -------------------------------------------------
import os
import dspy
from dspy.evaluate import SemanticF1
from datasets import load_dataset

# Load the WikiQA dataset
wikiqa = load_dataset("microsoft/wiki_qa")

# Split dataset into train, validation, and test sets
train_data = wikiqa["train"].select(range(100))
dev_data = wikiqa["validation"].select(range(50))
test_data = wikiqa["test"].select(range(10))

# Prepare the data for DSPy
def prepare_data(dataset):
    return [
        dspy.Example(
            question=example["question"],
            context=example["answer"],
            response=example["answer"]
        ).with_inputs("question")
        for example in dataset
    ]

trainset = prepare_data(train_data)
devset = prepare_data(dev_data)
testset = prepare_data(test_data)

# Configure DSPy
lm = dspy.LM('openai/gpt-4o-mini', 
             api_key=os.environ['OPENAI_API_KEY'])
dspy.configure(lm=lm)

# ------------------------------------------------- Code to be completed -------------------------------------------------
def init_variables():
    # Define evaluation metric
    # The basic metric for question answering tasks is the F1 score, dspy provides a modules implementing it.
    metric = ""

    # Define an evaluator
    evaluate = dspy.Evaluate(devset='', 
                            metric='', 
                            num_threads=24,
                            display_progress=True, 
                            display_table=2)

    # Use DSPy's Embedder and search functionalities
    embedder = dspy.Embedder('openai/text-embedding-3-small', 
                            dimensions=512)
    # Only select the answers from each example in the train set to initialize the corpus
    corpus = ['']
    search = dspy.retrievers.Embeddings(embedder="", 
                                        corpus="", 
                                        k=5)
    return metric, evaluate, search

def init_simple_rag(metric, evaluate, search) -> dspy.Module:
    # Define RAG (Retrieval-Augmented Generation) module
    class RAG(dspy.Module):
        def __init__(self, num_docs=5):
            self.num_docs = num_docs
            # here, you have to choose the module you want your RAG to use (ChainOfThought, ProgramOfThought or ReAct)
            # You can also provide context, the query and set an answer type
            # Example: dspy.Predict('sentence -> sentiment: bool')
            self.respond = ""

        def forward(self, question):
            context = search(question).passages
            return self.respond(context=context, question=question)

    rag = RAG()
    evaluate(rag)
    return rag
    

def init_optimized_rag(metric, evaluate) -> dspy.Module:
    # Now were are going to optimize the prompts of the RAG using an optimization pipeline
    # Initialize a tp object using Model Inference and Pipeline Optimization Version 2 module from dspy.
    # You need to add the metric and choose parameters like auto or num_threads.
    tp = dspy.MIPROv2()

    # Use the tp.compile function to start the training to create an optimized RAG.
    optimized_rag = ""
    evaluate(optimized_rag)

    # Save the optimized model
    optimized_rag.save("./optimized_rag.json")

    return optimized_rag


def main():
    # Enter a question
    question = ""
    
    metric, evaluate, search = init_variables()

    # Init simple rag
    rag = ""

    # Init optimized rag
    optimized_rag = ""

    simple_response = rag(question=question)
    optimized_response = optimized_rag(question=question)

    print(f"Question: {question}")
    print(f"Simple RAG response: {simple_response.response}")
    print(f"Optimized RAG response: {optimized_response.response}")


if __name__ == "__main__":
    main()

# --- --- --- PROVIDED --- --- ---
import pandas as pd
import numpy as np

def load_dataset():
    # Load dataset
    df = pd.read_json("task_01_out.json")

    # Remove useless columns
    df.drop(columns=['positive_is_positive', 'negative_is_negative'], inplace=True)

    # Specify columns to split
    pos_cols = ['positive', 'positive_embedding']
    neg_cols = ['negative', 'negative_embedding']
    split_cols = pos_cols + neg_cols

    # Find the remaining columns
    remaining_cols = [col for col in df.columns if col not in split_cols]

    # Make subsets
    df_pos = df[remaining_cols + pos_cols].rename(columns={'positive': 'label', 'positive_embedding': 'embedding'})
    df_neg = df[remaining_cols + neg_cols].rename(columns={'negative': 'label', 'negative_embedding': 'embedding'})

    # Concatenate the subsets
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    return df

# --- --- --- STUDENT CODE from task 04 --- --- ---

def get_similarity_scores(df, input_embedding, topk:int|None=None):
    """
    Compute similarity scores using dot product and sort rows from most similar to least similar.
    Returns all row if topk=None (by default), else return only the topk rows.

    Returns:
    pd.DataFrame: DataFrame with original index and similarity scores sorted in descending order.
    Optionanlly, return only the topk rows.
    """
    # Compute dot product similarities
    df['similarity'] = df['embedding'].apply(lambda x: np.dot(x, input_embedding))
    
    # Sort by similarity in descending order and return index and score
    sorted_df = df[['similarity']].sort_values(by='similarity', ascending=False).reset_index()

    # Topk
    if topk is not None:
        sorted_df = sorted_df.head(topk)

    return sorted_df


# --- --- --- STUDENT CODE 'First' --- --- ---
from openai import OpenAI
import dotenv
from typing import cast
from pathlib import Path
import json

env = dotenv.dotenv_values('.env')
api_key = cast(str, env["OPENAI_API_KEY"])
model_name = cast(str, env["OPENAI_MODEL_NAME"])
embedding_name = cast(str, env["OPENAI_EMBEDDING_MODEL_NAME"])

client = OpenAI(api_key=api_key)


df = load_dataset()
l=len(df)//2

file_path = Path("task_05_cache.json")

if file_path.exists():
    with open(file_path, 'r') as file:
        hyde_embeddings:list[dict] = cast(list[dict], json.load(file))
else:
    hyde_embeddings:list[dict] = []
    hyplist:list[str] = []
    
    # LLM - Hyde: generate an hypothetical passage
    for i in range(l):
        s = df.iloc[i]['source']
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": f"Write a very short text about {s}, emphasising both the positive and negative aspects. Be concise, no more than 1 or 2 sentences for each."}],
            temperature=0
        )
        response = completion.choices[0].message.content
        assert response is not None
        hyde_embeddings.append({"source": s, "hypothetical": response})
        hyplist.append(response)
        

    embs = client.embeddings.create(model=embedding_name, input=hyplist).data
    for r, e in zip(hyde_embeddings, embs):
        r["hypothetical_embedding"] = e.embedding
    
    with open(file_path, 'w') as file:
        json.dump(hyde_embeddings, file, indent=2)

# --- --- --- STUDENT CODE 'Then' --- --- ---

topk=5
l=len(df)//2
to_inspect = []
for i in range(l):
    source_embedding = hyde_embeddings[i]['hypothetical_embedding']
    result = get_similarity_scores(df, source_embedding, topk=topk)
    top2 = result.head(2)['index'].values
    diff = np.setdiff1d(top2, [i, i+l])
    if len(diff) > 0:
        to_inspect.append(i)
    neg_result = (result['index'] >= l).sum()
    print(f"{i}  source: {df.iloc[i]['source']:15} | other in top 2: {str(diff):8} | positive {topk-neg_result} | negative {neg_result}")


print()
print(f"Index to inspect: {to_inspect}")
print()

for i in to_inspect:
    source_embedding = df.iloc[i]['source_embedding']
    result = get_similarity_scores(df, source_embedding, topk=None)
    print(f"{i}  source: {df.iloc[i]['source']}")
    print(result)
    print()


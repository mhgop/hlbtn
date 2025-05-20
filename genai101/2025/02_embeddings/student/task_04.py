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

# --- --- --- STUDENT CODE --- --- ---
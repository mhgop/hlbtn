# --- --- --- PROVIDED CODE --- --- ---
import dotenv
from typing import cast
import json
import numpy as np

env = dotenv.dotenv_values('.env')
api_key = cast(str, env["OPENAI_API_KEY"])
embedding_name = cast(str, env["OPENAI_EMBEDDING_MODEL_NAME"])

with open('task_00_out.json', 'r') as f:
    data:list[dict] = list(json.load(f))

with open('task_01.json', 'r') as f:
    ref = json.load(f)
    
ref_positive = np.array(ref["positive_embedding"])
ref_negative = np.array(ref["negative_embedding"])

correct_counter = 0.0
total_counter = 0.0

# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---


    
# --- --- --- --- --- ---

print("--- --- ---")
print(f"Correct: {correct_counter}")
print(f"Total: {total_counter}")
print(f"Accuracy: {correct_counter/total_counter:.2%}")


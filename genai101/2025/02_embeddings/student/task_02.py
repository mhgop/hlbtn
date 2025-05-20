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
    
# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---
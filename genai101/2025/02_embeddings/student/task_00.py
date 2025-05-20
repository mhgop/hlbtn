# --- --- --- PROVIDED CODE --- --- ---
import dotenv
from typing import cast

env = dotenv.dotenv_values('.env')
api_key = cast(str, env["OPENAI_API_KEY"])
model_name = cast(str, env["OPENAI_MODEL_NAME"])
embedding_name = cast(str, env["OPENAI_EMBEDDING_MODEL_NAME"])

# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---
from openai import OpenAI
import json

client = OpenAI(api_key=api_key)

with open('task_00.json', 'r') as f:
    data:list[str] = json.load(f)


results:list[dict] = []

for source in data:
    # To complete...

    # A bit of feedback, not necessary for the task:
    print(f"Theme: {source:<15}")


with open('task_00_out.json', 'w') as f:
    json.dump(results, f, indent=2)
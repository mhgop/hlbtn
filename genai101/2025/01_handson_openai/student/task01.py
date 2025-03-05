# --- --- --- PROVIDED CODE --- --- ---
import dotenv
from typing import cast

env = dotenv.dotenv_values('.env')
api_key = cast(str, env["OPENAI_API_KEY"])
model_name = cast(str, env["OPENAI_MODEL_NAME"])

# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---


# --- --- --- TESTING CODE --- --- ---
import json

with open('task01.json') as json_file:
    config=json.load(json_file)

system_prompt = {"role":"system", "content":config["system"]}
context = [system_prompt]

while True:
    user_input = input("You: ")
    msg, context = call_with(user_input, context)
    print(f"AI: {msg}")
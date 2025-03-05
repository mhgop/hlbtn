# --- --- --- PROVIDED CODE --- --- ---
import dotenv
from typing import cast

env = dotenv.dotenv_values('.env')
api_key = cast(str, env["OPENAI_API_KEY"])
model_name = cast(str, env["OPENAI_MODEL_NAME"])

# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---
# define 'completion' variable in this section as the result of a call to the OpenAI API


# --- --- --- TESTING CODE --- --- ---

print(completion.choices[0].message.content)
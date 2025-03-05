# --- --- --- PROVIDED CODE --- --- ---
import dotenv
from typing import cast, Any

env = dotenv.dotenv_values('.env')
api_key = cast(str, env["OPENAI_API_KEY"])
model_name = cast(str, env["OPENAI_MODEL_NAME"])


# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---



# --- --- --- TESTING CODE --- --- ---
import sys

if __name__ == "__main__":

    try:
        st = sys.argv[1]
        print("Searching for:", st)
        r = call_with(st)
    except IndexError:
        print("Usage: program <search_term>")
        sys.exit(1)

    print("Final answer:", r)
    
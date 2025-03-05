# --- --- --- PROVIDED CODE --- --- ---
import dotenv
from typing import cast, Any

env = dotenv.dotenv_values('.env')
api_key = cast(str, env["OPENAI_API_KEY"])
model_name = cast(str, env["OPENAI_MODEL_NAME"])

# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---



# --- --- --- TESTING CODE --- --- ---

rootfolder = "assets/inputs"

if __name__ == "__main__":
    nb_retry = 3
    for i in range(2,13):
        # Retry loop
        for r in range(1, nb_retry+1):
            try:
                ipath = rootfolder + f"/{i}x{i}.jpeg"
                s = call_with(ipath)
                print(s)
                break # break out of retry loop
            except Exception as e:
                print(f"Error at {i}x{i} - try {r}")
        if r == nb_retry:
            print(f"Error at {i}x{i}: failed {nb_retry} times")
# --- --- --- PROVIDED CODE --- --- ---
import dotenv
from typing import cast, Any

env = dotenv.dotenv_values('.env')
api_key = cast(str, env["OPENAI_API_KEY"])
model_name = cast(str, env["OPENAI_MODEL_NAME"])

# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---



# --- --- --- TESTING CODE --- --- ---


def is_ascii(s):
    try:
        s.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


if __name__ == "__main__":
    txt = call_with("assets/inputs/screen_lesson.jpeg")
    print(txt)
    #
    assert is_ascii(txt), "The response should contain only ASCII characters"
    #
    assert "<h1>Introduction</h1>" in txt, "The response should contain '<h1>Introduction</h1>'"
    assert "<strong>Large Language Models</strong>" in txt, "The response should contain '<strong>Large Language Models</strong>'"
    assert '"Generative AI"' in txt, 'The response should contain "Generative AI"'
    #
    txt_low = txt.lower()
    assert "go to tasks" not in txt_low, "The response should not contain 'Go to tasks'"

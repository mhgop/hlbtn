# --- --- --- PROVIDED CODE --- --- ---
with open("paris_wiki.txt", "r") as f:
    text = f.read()

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
import tiktoken

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string for cl100k_base encoding."""
    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---

# --- Fixed Size Splitter
# TO DO

# --- Sliding Window Splitter
# TO DO

# --- Character Splitter - default separator
# TO DO

# --- Character Splitter - custom separator
# TO DO

# --- Recursive Character Splitter - default separator
# TO DO

# --- Recursive Character Splitter - custom separator
# TO DO
# --- --- --- PROVIDED CODE --- --- ---

from rank_bm25 import BM25Okapi
import sys
import numpy as np

with open("paris_wiki.txt", "r") as f:
    corpus = f.read().splitlines()
    
with open("task_07_stopwords.txt", "r") as f:
    stopwords = set(f.read().splitlines())

# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---

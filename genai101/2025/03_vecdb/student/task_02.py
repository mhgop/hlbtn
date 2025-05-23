from qdrant_client import models, QdrantClient
from qdrant_client.models import PointStruct, ScoredPoint
from fastembed import SparseTextEmbedding, SparseEmbedding

# --- --- --- Get embedding models

# Will download models on first run
splade_embedding_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")      # 532MB!
bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25", language="english")   # small


# --- --- --- New collection

client = QdrantClient(host="localhost", port=6333)
cname="sparse"

def create_collection():
    pass
#

create_collection()


# --- --- --- Load dataset

import pandas as pd
df = pd.read_json("../input.json")
texts = list(map(lambda triplet: "\n".join(triplet), zip(df["source"].tolist(), df["positive"].tolist(), df["negative"].tolist())))


# --- --- --- Sparse embeddings with SPLADE and BM25
 
def insert_data():
    pass
#

insert_data()

# --- --- --- Search

queries = [ "celestial", "astral" ]

def print_results(results:list[ScoredPoint]):
    for r in results:
        id = int(r.id)
        print(f"  Point ID: {id}, Score: {r.score}, Source: {df.iloc[id]['source']}")
        print(f"  Positive: {df.iloc[id]['positive']}")
        print(f"  Negative: {df.iloc[id]['negative']}")
        print()


for q in queries:
    print(f"SPLADE Query '{q}'")
    e = iter(splade_embedding_model.query_embed(q)).__next__()
    v = models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
    results = client.query_points(collection_name=cname, query=v, using="splade", limit=3).points
    print_results(results)
    print()

    print(f"BM25 Query '{q}'")
    e = iter(bm25_embedding_model.query_embed(q)).__next__()
    v = models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
    results = client.query_points(collection_name=cname, query=v, using="bm25").points
    print_results(results)
    print("\n")
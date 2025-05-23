from qdrant_client import models, QdrantClient
from qdrant_client.models import PointStruct, ScoredPoint, Prefetch, Fusion
from fastembed import SparseTextEmbedding, TextEmbedding


# --- --- --- Get embedding models

# Will download models on first run
dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2") # 90MB, good for demo but not best quality
bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25", language="english")   # small

test = list(dense_embedding_model.embed("This is a test"))[0]
dense_embedding_size = len(test)
print("Embedding size:", dense_embedding_size)


# --- --- --- New collection

client = QdrantClient(host="localhost", port=6333)
cname="hybrid"

def create_collection():
    pass
#

create_collection()


# --- --- --- Load dataset

import pandas as pd
df = pd.read_json("../input.json")
texts = list(map(lambda triplet: "\n".join(triplet), zip(df["source"].tolist(), df["positive"].tolist(), df["negative"].tolist())))


# --- --- --- Insert data: Dense and sparse embeddings

def insert_data():
    pass
#

insert_data()


# --- --- --- Search

queries = [ "body", "astral body" ]

N = 3

def print_results(pfx:str, results:list[ScoredPoint]):
    l = []
    for r in results:
        id = int(r.id)
        l.append(f"{r.score:0.3f} {id:2d} {df.iloc[id]['source']:<10}")
    print( f"{pfx:<15}: {' | '.join(l)}")
#


def mk_dense_prefetch(query:str, limit:int) -> Prefetch:
    ...
#


def mk_bm25_prefetch(query:str, limit:int) -> Prefetch:
    ...
#

def mk_prefetch(query:str, limit:int) -> list[Prefetch]:
    return [
        mk_dense_prefetch(query, limit),
        mk_bm25_prefetch(query, limit)
    ]
#
    
def hybrid_search(client, cname:str, prefetch:list[Prefetch], fusion:Fusion, limit:int) -> list[ScoredPoint]:
    return []
#


for q in queries:
    print(f"\n=== === === {q} === === ===")

    v = iter(dense_embedding_model.query_embed(q)).__next__()
    results = client.query_points(collection_name=cname, query=v, using="dense", limit=N).points
    print_results("Dense", results)

    e = iter(bm25_embedding_model.query_embed(q)).__next__()
    v = models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
    results = client.query_points(collection_name=cname, query=v, using="bm25", limit=N).points
    print_results("BM25", results)

    prefetch = mk_prefetch(q, limit=N)

    results = hybrid_search(client, cname, prefetch, Fusion.RRF, 2*N)
    print_results("RRF", results)

    results = hybrid_search(client, cname, prefetch, Fusion.DBSF, 2*N)
    print_results("DBSF", results)
#


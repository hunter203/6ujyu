from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import openai
import faiss
import numpy as np
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("aicpa_code_sections.json", "r") as f:
    code_data = json.load(f)

EMBEDDING_DIM = 1536
index = faiss.IndexFlatL2(EMBEDDING_DIM)
embeddings = []

for section in code_data:
    section_text = section["text"]
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=section_text
    )
    vector = np.array(response['data'][0]['embedding'], dtype='float32')
    embeddings.append(vector)

index.add(np.array(embeddings))

class Query(BaseModel):
    prompt: str

@app.post("/search")
def search_aicpa_code(query: Query):
    user_vector = openai.Embedding.create(
        model="text-embedding-3-small",
        input=query.prompt
    )['data'][0]['embedding']
    D, I = index.search(np.array([user_vector], dtype='float32'), k=5)
    return [
        {
            "section": code_data[idx]["section"],
            "title": code_data[idx]["title"],
            "text": code_data[idx]["text"][:500] + "...",
            "score": float(D[0][i])
        }
        for i, idx in enumerate(I[0])
    ]
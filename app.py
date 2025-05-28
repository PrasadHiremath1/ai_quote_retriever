import os
import sys
import types

import torch
torch.classes = types.SimpleNamespace()
sys.modules['torch.classes'] = torch.classes

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

from dotenv import load_dotenv
load_dotenv()  

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import replicate

os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

# Title
st.title("📚 Semantic Quote Finder (RAG-Based)")

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("quote_embedding_model")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("quote_metadata.csv")
    embeddings = np.load("quote_embeddings.npy")
    # Load FAISS index from saved file
    index = faiss.read_index("faiss_index.index")
    return df, embeddings, index

model = load_model()
df, embeddings, index = load_data()

def retrieve_quotes(query, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = df.iloc[indices[0]].copy()
    if np.max(distances[0]) != 0:
        results['similarity_score'] = 1 - distances[0] / np.max(distances[0])
    else:
        results['similarity_score'] = 1.0

    if isinstance(results.iloc[0]["tags"], str):
        results["tags"] = results["tags"].apply(eval)
    return results

def generate_response(query, retrieved_df):
    context = "\n".join(
        f"Quote: {row['quote']}\nAuthor: {row['author']}\nTags: {row['tags']}"
        for _, row in retrieved_df.iterrows()
    )

    prompt = f"""
You are a strict JSON-generating assistant.

User Query: "{query}"

You are given the following context containing quotes. Return only the quotes that best match the query.

Respond ONLY with a valid JSON array using the following format:
[
  {{
    "quote": "string",
    "author": "string",
    "tags": ["string", ...]
  }},
  ...
]

Do NOT include any explanation or commentary. Return only a JSON array.

Context:
{context}
"""

    output = replicate.run(
        "stability-ai/stablelm-tuned-alpha-7b:943c4afb4d0273cf1cf17c1070e182c903a9fe6b372df36b5447cf45935c42f2",
        input={
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9
        }
    )
    return "".join(output)

query = st.text_input(
    "🔍 Enter your quote-related query:",
    placeholder="e.g., motivational quotes about courage by women authors"
)

search_button = st.button("🔎 Search")

if search_button and query:
    with st.spinner("Retrieving and generating..."):
        retrieved_df = retrieve_quotes(query)
        json_response = generate_response(query, retrieved_df)

        st.subheader("🎯 Retrieved Quotes (Top 5):")
        for _, row in retrieved_df.iterrows():
            st.markdown(f"**{row['quote']}** — *{row['author']}*")
            st.markdown(f"**Tags**: `{', '.join(row['tags'])}`")
            st.markdown(f"**Similarity Score**: {row['similarity_score']:.2f}")
            st.markdown("---")

        st.subheader("📦 Structured JSON Output:")
        st.code(json_response, language="json")


# paste in .env REPLICATE_API_TOKEN=r8_QnpZp5ohlk3ts9s4SvBJAJMkEnr81ic3psh0N 

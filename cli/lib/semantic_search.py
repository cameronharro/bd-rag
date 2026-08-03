import os

import numpy as np
from sentence_transformers import SentenceTransformer
from lib.search_utils import Movie, MOVIE_EMBEDDINGS_PATH, load_movies

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        text = text.strip()
        if text == "":
            raise ValueError("empty text")
        encoding = self.model.encode([text])
        return encoding[0]
    
    def populate_documents(self, documents):
        self.documents = documents
        for _, movie in enumerate(documents):
            self.document_map[movie["id"]] = movie

    def load_or_create_embeddings(self, documents):
        self.populate_documents(documents)
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def build_embeddings(self, documents: list[Movie]):
        self.populate_documents(documents)
        strs: list[str] = [""] * len(documents)
        for i, movie in enumerate(documents):
            strs[i] = f"{movie["title"]}: {movie["description"]}"
        self.embeddings = self.model.encode(strs, show_progress_bar=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings


def verify_model():
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")

def embed_text(text: str):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    sem_search = SemanticSearch()
    movies = load_movies()
    embeddings = sem_search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

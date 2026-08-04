import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from torch import sort
from lib.search_utils import Movie, MOVIE_EMBEDDINGS_PATH, SearchResult, format_search_result, load_movies

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
    
    def search(self, query: str, limit: int) -> list[SearchResult]:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        result: list[tuple[float, Movie]] = []
        for i, doc_embedding in enumerate(self.embeddings):
            result.append((cosine_similarity(query_embedding, doc_embedding), self.documents[i]))
        result.sort(reverse=True, key=lambda item: item[0])
        return [format_search_result(x[1]["id"], x[1]["title"], x[1]["description"], x[0]) for x in result[:limit]]

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

def embed_query_text(query):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search(query: str, limit: int):
    sem_search = SemanticSearch()
    movies = load_movies()
    sem_search.load_or_create_embeddings(movies)
    results = sem_search.search(query, limit)
    for i, result in enumerate(results):
        print(f"{i + 1}. {result["title"]} (score: {result["score"]})\n  {result["document"]}\n")

def chunk(text: str, chunk_size: int, overlap: int):
    text = text.strip()
    words = text.split(" ")
    result: list[str] = []
    chunk_start = 0
    while chunk_start < len(words):
        chunk_length = min(chunk_size, len(words) - chunk_start)
        expanded_start = max(0, chunk_start - overlap)
        expanded_end = min(len(words), chunk_start+chunk_length+overlap)
        chunk = " ".join(words[expanded_start:expanded_end])
        result.append(chunk)
        chunk_start += chunk_length

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(result):
        print(f"{i + 1}. {chunk}")

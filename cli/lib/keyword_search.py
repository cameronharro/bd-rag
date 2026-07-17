import math
import os
import pickle
import string
from collections import defaultdict
from typing import Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    STOPWORDS_PATH,
    Movie,
    load_movies,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: defaultdict[str, set[int]] = defaultdict(set[int])
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies: defaultdict[int, Counter[str]] = defaultdict(Counter[str])
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id].get(term, 0)

    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[term])
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[term])
        return math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)

    def get_tfidf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def load(self):
        def get_obj(path: str):
            if not os.path.isfile(path):
                raise Exception(f"No file at {path}")
            with open(path, "rb") as file:
                return pickle.load(file)
        self.index = get_obj(self.index_path)
        self.docmap = get_obj(self.docmap_path)
        self.term_frequencies = get_obj(self.term_frequencies_path)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as e:
        print(e)
        return []
    query_tokens = tokenize_text(query)
    results = []
    for q in query_tokens:
        for id in idx.get_documents(q):
            results.append(idx.docmap[id])
        if len(results) >= limit:
            break

    return results[:min(len(results), limit)]


def tf_command(doc_id: int, term: str):
    term = get_single_token(term)
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as e:
        print(e)
        return
    print(idx.get_tf(doc_id, term))


def idf_command(term: str):
    token = ""
    idx = InvertedIndex()
    try:
        token = get_single_token(term)
        idx.load()
    except Exception as e:
        print(e)
        return
    print(f"Inverse document frequency of '{term}': {idx.get_idf(token):.2f}")


def bm25idf_command(term: str):
    token = ""
    idx = InvertedIndex()
    try:
        token = get_single_token(term)
        idx.load()
    except Exception as e:
        print(e)
        return
    print(f"BM25 IDF score of '{term}': {idx.get_bm25idf(token):.2f}")


def tfidf_command(doc_id: int, term: str):
    token = ""
    idx = InvertedIndex()
    try:
        token = get_single_token(term)
        idx.load()
    except Exception as e:
        print(e)
        return
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {idx.get_tfidf(doc_id, token):.2f}")


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]


STOPWORDS = load_stopwords()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    filtered_words = []
    for word in valid_tokens:
        if word not in STOPWORDS:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def get_single_token(text: str) -> str:
    tokens = tokenize_text(text)
    if len(tokens) != 1:
        raise Exception(f"{text} is more than one token")
    return tokens[0]

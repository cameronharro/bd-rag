import argparse
import json
import os
import pickle
import string
from typing import TypedDict
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
def preprocess_text(text: str, stopwords: list[str] = []) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans({x: "" for x in string.punctuation}))
    tokens = text.split(" ")
    return [stemmer.stem(t) for t in tokens if t not in stopwords]

def tokens_match(text_tokens: list[str], query_tokens: list[str]) -> bool:
    for t in text_tokens:
        for q in query_tokens:
            if q in t:
                return True
    return False

Movie = TypedDict("Movie", {"id": int, "title": str, "description": str})
def load_movies(path: str) -> list[Movie]:
    movie_file = open(path)
    return json.load(movie_file)["movies"]

def load_stopwords(path) -> list[str]:
    text = open(path).read().replace("\n", " ")
    return preprocess_text(text)
STOPWORDS = load_stopwords("./data/stopwords.txt")

class InvertedIndex:
    index: dict[str, set[int]] = {}
    docmap: dict[int, str] = {}
    
    def __add_document(self, doc_id: int, text: str):
        self.docmap[doc_id] = text
        tokens = preprocess_text(text, STOPWORDS)
        for token in tokens:
            if self.index.get(token) == None:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        if self.index.get(term) == None:
            return []
        return sorted(self.index[term])

    def build(self):
        movies = load_movies("./data/movies.json")
        for movie in movies:
            self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}")

    def save(self):
        CACHE_PATH = "cache"
        if not os.path.isdir(CACHE_PATH):
            os.mkdir(CACHE_PATH)
        with open(CACHE_PATH + "/index.pkl", "wb") as index_file:
            pickle.dump(self.index, index_file)
        with open(CACHE_PATH + "/docmap.pkl", "wb") as docmap_file:
            pickle.dump(self.index, docmap_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Parse Movie data for searching")

    args = parser.parse_args()

    match args.command:
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case "search":
            movies = load_movies("./data/movies.json")
            query = args.query
            query_tokens = preprocess_text(query, STOPWORDS)
            matches = lambda text: tokens_match(preprocess_text(text, STOPWORDS), query_tokens)
            print(f"Searching for: {query}")
            matched_movies = [movie["title"] for movie in movies if matches(movie["title"])][:5]
            for i, title in enumerate(matched_movies):
                print(f"{i+1}: {title}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

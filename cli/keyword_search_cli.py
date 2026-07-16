import argparse
import json
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            movies = load_movies("./data/movies.json")
            stopwords = load_stopwords("./data/stopwords.txt")
            query = args.query
            query_tokens = preprocess_text(query, stopwords)
            matches = lambda text: tokens_match(preprocess_text(text, stopwords), query_tokens)
            print(f"Searching for: {query}")
            matched_movies = [movie["title"] for movie in movies if matches(movie["title"])][:5]
            for i, title in enumerate(matched_movies):
                print(f"{i+1}: {title}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

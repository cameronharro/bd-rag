import argparse
import json
import string
from typing import TypedDict

def preprocess_text(text: str, stopwords: list[str] = []) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans({x: "" for x in string.punctuation}))
    tokens = text.split(" ")
    return [t for t in tokens if t not in stopwords]

def tokens_match(text_tokens: list[str], query_tokens: list[str]) -> bool:
    for t in text_tokens:
        for q in query_tokens:
            if q in t:
                return True
    return False

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            movie_file_path = "./data/movies.json"
            movie_file = open(movie_file_path)
            Movie = TypedDict("Movie", {"id": int, "title": str, "description": str})
            movies: list[Movie] = json.load(movie_file)["movies"]

            stopwords = open("./data/stopwords.txt").read().replace("\n", " ")
            stopwords = preprocess_text(stopwords)
            def matches(text: str, query: str) -> bool:
                text_tokens = preprocess_text(text, stopwords)
                query_tokens = preprocess_text(query, stopwords)
                return tokens_match(text_tokens, query_tokens)
            query = args.query
            print(f"Searching for: {query}")
            matched_movies = [movie["title"] for movie in movies if matches(movie["title"], query)][:5]
            for i, title in enumerate(matched_movies):
                print(f"{i}: {title}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

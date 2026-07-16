import argparse
import json
import string
from typing import TypedDict

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
            def matches(text: str, query: str) -> bool:
                text = text.lower()
                text = text.translate(str.maketrans({x: "" for x in string.punctuation}))
                return query in text
            query = args.query
            print(f"Searching for: {query}")
            matched_movies = [movie["title"] for movie in movies if matches(movie["title"], query)][:5]
            for i, title in enumerate(matched_movies):
                print(f"{i}: {title}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

import argparse

from lib.search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT
from lib.keyword_search import bm25idf_command, bm25tf_command, bmd25search_command, build_command, idf_command, search_command, tf_command, tfidf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get the frequency of a term in a given document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="The term to retrieve")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency of a given term")
    idf_parser.add_argument("term", type=str, help="The term to retrieve")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get the BM25 inverse document frequency of a given term")
    bm25idf_parser.add_argument("term", type=str, help="The term to retrieve")

    bm25tf_parser = subparsers.add_parser("bm25tf", help="Get the BM25 frequency of a term in a given document")
    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="The term to retrieve")
    bm25tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 B parameter")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the frequency that a term appears in a given document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="The term to retrieve")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Search limit")

    args = parser.parse_args()

    try:
        match args.command:
            case "build":
                print("Building inverted index...")
                build_command()
                print("Inverted index built successfully.")
                return
            case "search":
                print("Searching for:", args.query)
                results = search_command(args.query)
                for i, res in enumerate(results, 1):
                    print(f"{i}. ({res['id']}) {res['title']}")
                return
            case "tf":
                tf_command(args.doc_id, args.term)
                return
            case "idf":
                idf_command(args.term)
                return
            case "bm25idf":
                bm25idf_command(args.term)
                return
            case "bm25tf":
                bm25tf_command(args.doc_id, args.term, args.k1, args.b)
                return
            case "tfidf":
                tfidf_command(args.doc_id, args.term)
                return
            case "bm25search":
                bmd25search_command(args.query, args.limit)
            case _:
                parser.print_help()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()

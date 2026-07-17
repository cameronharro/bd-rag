import argparse

from lib.keyword_search import build_command, idf_command, search_command, tf_command, tfidf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get the frequency that a term appears in a given document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="The term to retrieve")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency of a given term")
    idf_parser.add_argument("term", type=str, help="The term to retrieve")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the frequency that a term appears in a given document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="The term to retrieve")

    args = parser.parse_args()

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
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
